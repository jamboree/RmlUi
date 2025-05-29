#include "RmlUi_Renderer_VX.h"
#include <RmlUi/Core/Core.h>
#include <RmlUi/Core/DecorationTypes.h>
#include <RmlUi/Core/FileInterface.h>
#include <RmlUi/Core/Log.h>
#include "RmlUi_VX/ShadersCompiledSPV.h"

struct StagingBuffer {
    vma::Allocator m_Allocator;
    // Result
    vk::Buffer m_Buffer;
    vma::Allocation m_Allocation;

    void* Alloc(vk::DeviceSize size) {
        vk::BufferCreateInfo bufferInfo;
        bufferInfo.setSize(size);
        bufferInfo.setUsage(vk::BufferUsageFlagBits::bTransferSrc);
        bufferInfo.setSharingMode(vk::SharingMode::eExclusive);

        vma::AllocationCreateInfo allocationInfo;
        allocationInfo.setUsage(vma::MemoryUsage::eAuto);
        allocationInfo.setFlags(
            vma::AllocationCreateFlagBits::bMapped |
            vma::AllocationCreateFlagBits::bHostAccessSequentialWrite);

        vma::AllocationInfo allocInfo;
        m_Buffer = m_Allocator
                       .createBuffer(bufferInfo, allocationInfo, &m_Allocation,
                                     &allocInfo)
                       .get();

        return allocInfo.getMappedData();
    }

    ~StagingBuffer() {
        if (m_Buffer) {
            m_Allocator.destroyBuffer(m_Buffer, m_Allocation);
        }
    }
};

struct Renderer_VX::GeometryResource {
    uint32_t m_VertexCount;
    uint32_t m_IndexCount;
    vk::Buffer m_Buffer;
    vma::Allocation m_Allocation;

    void Draw(vx::CommandBuffer commandBuffer) const {
        const vk::DeviceSize offset = 0;
        commandBuffer.cmdBindVertexBuffers(0, 1, &m_Buffer, &offset);
        commandBuffer.cmdBindIndexBuffer(m_Buffer,
                                         m_VertexCount * sizeof(Rml::Vertex),
                                         vk::IndexType::eUint32);
        commandBuffer.cmdDrawIndexed(m_IndexCount, 1, 0, 0, 0);
    }
};

struct Renderer_VX::TextureResource {
    vk::Image m_Image;
    vk::ImageView m_ImageView;
    vma::Allocation m_Allocation;
};

struct Renderer_VX::ShaderResource {
    vk::Buffer m_Buffer;
    vma::Allocation m_Allocation;
};

struct VsInput {
    Rml::Matrix4f transform;
    Rml::Vector2f translate;
};

#define FIELD(s, m) offsetof(s, m), sizeof(s::m)

struct Renderer_VX::TextureDescriptorSet {
    static constexpr vk::ShaderStageFlags Stages =
        vk::ShaderStageFlagBits::bFragment;

    VX_BINDING(0, vx::CombinedImageSamplerDescriptor, Stages) tex;
};

struct Renderer_VX::GradientDescriptorSet {
    static constexpr vk::ShaderStageFlags Stages =
        vk::ShaderStageFlagBits::bFragment;

    VX_BINDING(0, vx::UniformBufferDescriptor, Stages) uniform;
};

Renderer_VX::Renderer_VX() = default;
Renderer_VX::~Renderer_VX() = default;

bool Renderer_VX::Init(RenderContext_VX& context,
                       vk::PipelineRenderingCreateInfo& renderingInfo) {
    m_Context = &context;

    const auto device = m_Context->GetDevice();

    vk::SamplerCreateInfo samplerInfo;
    samplerInfo.setMagFilter(vk::Filter::eLinear);
    samplerInfo.setMinFilter(vk::Filter::eLinear);
    samplerInfo.setAddressModeU(vk::SamplerAddressMode::eRepeat);
    samplerInfo.setAddressModeV(vk::SamplerAddressMode::eRepeat);
    samplerInfo.setAddressModeW(vk::SamplerAddressMode::eRepeat);
    m_Sampler = device.createSampler(samplerInfo).get();

    InitPipelineLayouts();
    InitPipelines(renderingInfo);

    return true;
}

void Renderer_VX::Shutdown() {
    const auto device = m_Context->GetDevice();

    if (m_Sampler) {
        device.destroySampler(m_Sampler);
    }
    if (m_ClipPipeline) {
        device.destroyPipeline(m_ClipPipeline);
    }
    if (m_ColorPipeline) {
        device.destroyPipeline(m_ColorPipeline);
    }
    if (m_TexturePipeline) {
        device.destroyPipeline(m_TexturePipeline);
    }
    if (m_GradientPipeline) {
        device.destroyPipeline(m_GradientPipeline);
    }
    if (m_BasicPipelineLayout) {
        device.destroyPipelineLayout(m_BasicPipelineLayout);
    }
    if (m_TexturePipelineLayout) {
        device.destroyPipelineLayout(m_TexturePipelineLayout);
    }
    if (m_GradientPipelineLayout) {
        device.destroyPipelineLayout(m_GradientPipelineLayout);
    }
    if (m_TextureDescriptorSetLayout) {
        device.destroyDescriptorSetLayout(m_TextureDescriptorSetLayout);
    }
    if (m_GradientDescriptorSetLayout) {
        device.destroyDescriptorSetLayout(m_GradientDescriptorSetLayout);
    }
}

static Rml::Matrix4f Project(vk::Extent2D extent,
                             const Rml::Matrix4f& transform) {
    auto projection = Rml::Matrix4f::ProjectOrtho(
        0.0f, float(extent.getWidth()), float(extent.getHeight()), 0.0f, -10000,
        10000);

    // https://matthewwellings.com/blog/the-new-vulkan-coordinate-system/
    Rml::Matrix4f correction_matrix;
    correction_matrix.SetColumns(Rml::Vector4f(1.0f, 0.0f, 0.0f, 0.0f),
                                 Rml::Vector4f(0.0f, -1.0f, 0.0f, 0.0f),
                                 Rml::Vector4f(0.0f, 0.0f, 0.5f, 0.0f),
                                 Rml::Vector4f(0.0f, 0.0f, 0.5f, 1.0f));

    return correction_matrix * projection * transform;
}

void Renderer_VX::BeginFrame(vx::CommandBuffer commandBuffer, uint32_t frame) {
    m_CommandBuffer = commandBuffer;
    m_FrameNumber = frame;
    m_StencilRef = 1;
    m_EnableScissor = false;

    const auto extent = m_Context->GetFrameExtent();
    m_Scissor.setOffset({});
    m_Scissor.setExtent(extent);
    m_CommandBuffer.cmdSetScissor(0, 1, &m_Scissor);

    const auto transform = Project(extent, Rml::Matrix4f::Identity());
    m_CommandBuffer.cmdPushConstants(m_BasicPipelineLayout,
                                     vk::ShaderStageFlagBits::bVertex,
                                     FIELD(VsInput, transform), &transform);

    m_CommandBuffer.cmdSetStencilTestEnable(false);
}

void Renderer_VX::EndFrame() { m_CommandBuffer = {}; }

void Renderer_VX::ResetFrame(uint32_t frame) {
    const uint8_t useFlag = 2u << frame;
    m_GeometryResources.ReleaseAllUse(*this, useFlag);
    m_TextureResources.ReleaseAllUse(*this, useFlag);
    m_ShaderResources.ReleaseAllUse(*this, useFlag);
}

Rml::CompiledGeometryHandle
Renderer_VX::CompileGeometry(Rml::Span<const Rml::Vertex> vertices,
                             Rml::Span<const int> indices) {
    const auto allocator = m_Context->GetAllocator();

    const auto [index, g] = m_GeometryResources.Allocate();
    g->m_VertexCount = uint32_t(vertices.size());
    g->m_IndexCount = uint32_t(indices.size());

    const auto vertexBytes = g->m_VertexCount * sizeof(Rml::Vertex);
    const auto indexBytes = g->m_IndexCount * sizeof(uint32_t);
    vk::BufferCreateInfo bufferInfo;
    bufferInfo.setSize(vertexBytes + indexBytes);
    bufferInfo.setUsage(vk::BufferUsageFlagBits::bVertexBuffer |
                        vk::BufferUsageFlagBits::bIndexBuffer);
    bufferInfo.setSharingMode(vk::SharingMode::eExclusive);

    vma::AllocationCreateInfo allocationInfo;
    allocationInfo.setFlags(
        vma::AllocationCreateFlagBits::bMapped |
        vma::AllocationCreateFlagBits::bHostAccessSequentialWrite);
    allocationInfo.setUsage(vma::MemoryUsage::eAutoPreferDevice);

    vma::AllocationInfo allocInfo;
    g->m_Buffer = allocator
                      .createBuffer(bufferInfo, allocationInfo,
                                    &g->m_Allocation, &allocInfo)
                      .get();

    auto p = static_cast<uint8_t*>(allocInfo.getMappedData());
    std::memcpy(p, vertices.data(), vertexBytes);
    p += vertexBytes;
    std::memcpy(p, indices.data(), indexBytes);

    return ~index;
}

void Renderer_VX::RenderGeometry(Rml::CompiledGeometryHandle geometry,
                                 Rml::Vector2f translation,
                                 Rml::TextureHandle texture) {
    const uint8_t useFlag = 2u << m_FrameNumber;
    const auto g = m_GeometryResources.Use(~geometry, useFlag);
    auto pipeline = m_ColorPipeline;
    auto pipelineLayout = m_BasicPipelineLayout;
    if (texture) {
        pipeline = m_TexturePipeline;
        pipelineLayout = m_TexturePipelineLayout;
        const auto t = m_TextureResources.Use(~texture, useFlag);
        const vx::DescriptorSet<TextureDescriptorSet> descriptorSet;
        m_CommandBuffer.cmdPushDescriptorSetKHR(
            vk::PipelineBindPoint::eGraphics, pipelineLayout, 0,
            {descriptorSet->tex = vx::CombinedImageSamplerDescriptor(
                 m_Sampler, t->m_ImageView,
                 vk::ImageLayout::eShaderReadOnlyOptimal)});
    }
    m_CommandBuffer.cmdBindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
    m_CommandBuffer.cmdPushConstants(pipelineLayout,
                                     vk::ShaderStageFlagBits::bVertex,
                                     FIELD(VsInput, translate), &translation);
    g->Draw(m_CommandBuffer);
}

void Renderer_VX::Destroy(GeometryResource& g) {
    const auto allocator = m_Context->GetAllocator();
    allocator.destroyBuffer(g.m_Buffer, g.m_Allocation);
}

void Renderer_VX::ReleaseGeometry(Rml::CompiledGeometryHandle geometry) {
    RMLUI_ASSERT(geometry);
    m_GeometryResources.Release(*this, ~geometry);
}

struct TGAHeader {
    uint8_t idLength;
    uint8_t colourMapType;
    uint8_t dataType;
    uint8_t colourMap[5];
    short int xOrigin;
    short int yOrigin;
    short int width;
    short int height;
    uint8_t bitsPerPixel;
    uint8_t imageDescriptor;
};

Rml::TextureHandle Renderer_VX::LoadTexture(Rml::Vector2i& texture_dimensions,
                                            const Rml::String& source) {
    Rml::FileInterface* file_interface = Rml::GetFileInterface();
    Rml::FileHandle file_handle = file_interface->Open(source);
    if (!file_handle) {
        return false;
    }

    file_interface->Seek(file_handle, 0, SEEK_END);
    size_t buffer_size = file_interface->Tell(file_handle);
    file_interface->Seek(file_handle, 0, SEEK_SET);

    if (buffer_size <= sizeof(TGAHeader)) {
        Rml::Log::Message(Rml::Log::LT_ERROR,
                          "Texture file size is smaller than TGAHeader, file "
                          "is not a valid TGA image.");
        file_interface->Close(file_handle);
        return false;
    }

    std::unique_ptr<uint8_t[]> buffer(new uint8_t[buffer_size]);
    file_interface->Read(buffer.get(), buffer_size, file_handle);
    file_interface->Close(file_handle);

    TGAHeader header;
    std::memcpy(&header, buffer.get(), sizeof(TGAHeader));

    int color_mode = header.bitsPerPixel / 8;
    const size_t image_size =
        header.width * header.height * 4; // We always make 32bit textures

    if (header.dataType != 2) {
        Rml::Log::Message(Rml::Log::LT_ERROR,
                          "Only 24/32bit uncompressed TGAs are supported.");
        return false;
    }

    // Ensure we have at least 3 colors
    if (color_mode < 3) {
        Rml::Log::Message(Rml::Log::LT_ERROR,
                          "Only 24 and 32bit textures are supported.");
        return false;
    }

    auto image_src = buffer.get() + sizeof(TGAHeader);
    StagingBuffer stagingBuffer{m_Context->GetAllocator()};

    auto image_dest = static_cast<uint8_t*>(stagingBuffer.Alloc(image_size));

    const bool topDown = header.imageDescriptor & 32;
    auto srcRowStep = header.width * color_mode;
    if (!topDown) {
        image_src += (header.height - 1) * srcRowStep;
        srcRowStep = -srcRowStep;
    }

    // Targa is BGR, swap to RGB, flip Y axis, and convert to premultiplied
    // alpha.
    for (int y = 0; y < header.height; ++y) {
        auto src = image_src;
        for (int x = 0; x < header.width; ++x) {
            uint8_t rgba[4] = {src[2], src[1], src[0], 255};
            if (color_mode == 4) {
                const auto alpha = src[3];
                for (int i = 0; i != 3; ++i) {
                    rgba[i] = uint8_t((rgba[i] * alpha) / 255);
                }
                rgba[3] = alpha;
            }
            std::memcpy(image_dest, rgba, 4);
            image_dest += 4;
            src += color_mode;
        }
        image_src += srcRowStep;
    }

    texture_dimensions.x = header.width;
    texture_dimensions.y = header.height;

    return CreateTexture(stagingBuffer.m_Buffer, texture_dimensions);
}

Rml::TextureHandle
Renderer_VX::GenerateTexture(Rml::Span<const Rml::byte> source_data,
                             Rml::Vector2i source_dimensions) {
    StagingBuffer stagingBuffer{m_Context->GetAllocator()};
    std::memcpy(stagingBuffer.Alloc(source_data.size()), source_data.data(),
                source_data.size());
    return CreateTexture(stagingBuffer.m_Buffer, source_dimensions);
}

void Renderer_VX::Destroy(TextureResource& t) {
    const auto device = m_Context->GetDevice();
    const auto allocator = m_Context->GetAllocator();
    device.destroyImageView(t.m_ImageView);
    allocator.destroyImage(t.m_Image, t.m_Allocation);
}

void Renderer_VX::ReleaseTexture(Rml::TextureHandle texture) {
    RMLUI_ASSERT(texture);
    m_TextureResources.Release(*this, ~texture);
}

void Renderer_VX::EnableScissorRegion(bool enable) {
    m_EnableScissor = enable;
    if (!m_EnableScissor) {
        m_Scissor.setOffset({});
        m_Scissor.setExtent(m_Context->GetFrameExtent());
        m_CommandBuffer.cmdSetScissor(0, 1, &m_Scissor);
    }
}

void Renderer_VX::SetScissorRegion(Rml::Rectanglei region) {
    if (m_EnableScissor) {
        m_Scissor.setOffset(vk::Offset2D(region.Left(), region.Top()));
        m_Scissor.setExtent(vk::Extent2D(region.Width(), region.Height()));
        m_CommandBuffer.cmdSetScissor(0, 1, &m_Scissor);
    }
}

void Renderer_VX::SetTransform(const Rml::Matrix4f* transform) {
    const auto extent = m_Context->GetFrameExtent();
    const auto matrix =
        Project(extent, transform ? *transform : Rml::Matrix4f::Identity());
    m_CommandBuffer.cmdPushConstants(m_BasicPipelineLayout,
                                     vk::ShaderStageFlagBits::bVertex,
                                     FIELD(VsInput, transform), &matrix);
}

void Renderer_VX::EnableClipMask(bool enable) {
    m_CommandBuffer.cmdSetStencilTestEnable(enable);
}

void Renderer_VX::RenderToClipMask(Rml::ClipMaskOperation operation,
                                   Rml::CompiledGeometryHandle geometry,
                                   Rml::Vector2f translation) {
    const uint8_t useFlag = 2u << m_FrameNumber;
    const auto g = m_GeometryResources.Use(~geometry, useFlag);

    bool clearStencil = false;
    auto stencilPassOp = vk::StencilOp::eReplace;
    switch (operation) {
    case Rml::ClipMaskOperation::Set:
        clearStencil = true;
        m_StencilRef = 1;
        break;
    case Rml::ClipMaskOperation::SetInverse:
        clearStencil = true;
        m_StencilRef = 0;
        break;
    case Rml::ClipMaskOperation::Intersect:
        stencilPassOp = vk::StencilOp::eIncrementAndClamp;
        ++m_StencilRef;
        break;
    }

    if (clearStencil) {
        vk::ClearAttachment clearAttachment;
        clearAttachment.setAspectMask(vk::ImageAspectFlagBits::bStencil);
        clearAttachment.setClearValue(
            {.depthStencil = vk::ClearDepthStencilValue{1.0f, 0}});
        vk::ClearRect clearRect;
        clearRect.setLayerCount(1);
        clearRect.setRect(m_Scissor);
        m_CommandBuffer.cmdClearAttachments(1, &clearAttachment, 1, &clearRect);
    }

    m_CommandBuffer.cmdBindPipeline(vk::PipelineBindPoint::eGraphics,
                                    m_ClipPipeline);
    m_CommandBuffer.cmdPushConstants(m_BasicPipelineLayout,
                                     vk::ShaderStageFlagBits::bVertex,
                                     FIELD(VsInput, translate), &translation);
    m_CommandBuffer.cmdSetStencilOp(
        vk::StencilFaceFlagBits::eFrontAndBack, vk::StencilOp::eKeep,
        stencilPassOp, vk::StencilOp::eKeep, vk::CompareOp::eAlways);

    g->Draw(m_CommandBuffer);

    m_CommandBuffer.cmdSetStencilReference(
        vk::StencilFaceFlagBits::eFrontAndBack, m_StencilRef);
}

inline Rml::Colourf ConvertToColorf(Rml::ColourbPremultiplied c0) {
    return {c0[0] / 255.f, c0[1] / 255.f, c0[2] / 255.f, c0[3] / 255.f};
}

struct GradientUniform {
    static constexpr int MAX_NUM_STOPS = 16;
    enum { LINEAR = 0, RADIAL = 1, CONIC = 2, REPEATING = 1 };

    // one of the above definitions
    int func;
    // linear: starting point, radial: center, conic: center
    int numStops;
    alignas(8) Rml::Vector2f pos;
    // linear: vector to ending point, radial: 2d curvature (inverse radius),
    // conic: angled unit vector
    alignas(8) Rml::Vector2f vec;
    // normalized, 0 -> starting point, 1 -> ending point
    float stopPositions[MAX_NUM_STOPS];
    alignas(16) Rml::Colourf stopColors[MAX_NUM_STOPS];

    void ApplyColorStopList(const Rml::Dictionary& parameters) {
        const auto it = parameters.find("color_stop_list");
        RMLUI_ASSERT(it != parameters.end() &&
                     it->second.GetType() == Rml::Variant::COLORSTOPLIST);
        const auto& color_stop_list =
            it->second.GetReference<Rml::ColorStopList>();
        numStops = Rml::Math::Min((int)color_stop_list.size(), MAX_NUM_STOPS);
        for (int i = 0; i != numStops; ++i) {
            const auto& stop = color_stop_list[i];
            RMLUI_ASSERT(stop.position.unit == Rml::Unit::NUMBER);
            stopPositions[i] = stop.position.number;
            stopColors[i] = ConvertToColorf(stop.color);
        }
    }

    vk::DeviceSize GetUsedSize() const {
        return offsetof(GradientUniform, stopColors) +
               sizeof(Rml::Colourf) * numStops;
    }
};

Rml::CompiledShaderHandle
Renderer_VX::CompileShader(const Rml::String& name,
                           const Rml::Dictionary& parameters) {
    GradientUniform uniform;
    if (name == "linear-gradient") {
        uniform.func = GradientUniform::LINEAR << 1;
        uniform.pos = Rml::Get(parameters, "p0", Rml::Vector2f(0.f));
        uniform.vec =
            Rml::Get(parameters, "p1", Rml::Vector2f(0.f)) - uniform.pos;
    } else if (name == "radial-gradient") {
        uniform.func = GradientUniform::RADIAL << 1;
        uniform.pos = Rml::Get(parameters, "center", Rml::Vector2f(0.f));
        uniform.vec = Rml::Vector2f(1.f) /
                      Rml::Get(parameters, "radius", Rml::Vector2f(1.f));
    } else if (name == "conic-gradient") {
        uniform.func = GradientUniform::CONIC << 1;
        uniform.pos = Rml::Get(parameters, "center", Rml::Vector2f(0.f));
        const float angle = Rml::Get(parameters, "angle", 0.f);
        uniform.vec = {Rml::Math::Cos(angle), Rml::Math::Sin(angle)};
    } else {
        Rml::Log::Message(Rml::Log::LT_WARNING, "Unsupported shader type '%s'.",
                          name.c_str());
        return {};
    }
    if (Rml::Get(parameters, "repeating", false))
        uniform.func |= GradientUniform::REPEATING;
    uniform.ApplyColorStopList(parameters);

    const auto allocator = m_Context->GetAllocator();

    const auto [index, s] = m_ShaderResources.Allocate();

    const auto uniformSize = uniform.GetUsedSize();
    vk::BufferCreateInfo bufferInfo;
    bufferInfo.setSize(uniformSize);
    bufferInfo.setUsage(vk::BufferUsageFlagBits::bUniformBuffer);
    bufferInfo.setSharingMode(vk::SharingMode::eExclusive);

    vma::AllocationCreateInfo allocationInfo;
    allocationInfo.setFlags(
        vma::AllocationCreateFlagBits::bMapped |
        vma::AllocationCreateFlagBits::bHostAccessSequentialWrite);
    allocationInfo.setUsage(vma::MemoryUsage::eAutoPreferDevice);

    vma::AllocationInfo allocInfo;
    s->m_Buffer = allocator
                      .createBuffer(bufferInfo, allocationInfo,
                                    &s->m_Allocation, &allocInfo)
                      .get();

    std::memcpy(allocInfo.getMappedData(), &uniform, uniformSize);

    return ~index;
}

void Renderer_VX::RenderShader(Rml::CompiledShaderHandle shader,
                               Rml::CompiledGeometryHandle geometry,
                               Rml::Vector2f translation,
                               Rml::TextureHandle /*texture*/) {
    const uint8_t useFlag = 2u << m_FrameNumber;
    const auto s = m_ShaderResources.Use(~shader, useFlag);
    const auto g = m_GeometryResources.Use(~geometry, useFlag);

    const vx::DescriptorSet<GradientDescriptorSet> descriptorSet;
    m_CommandBuffer.cmdPushDescriptorSetKHR(
        vk::PipelineBindPoint::eGraphics, m_GradientPipelineLayout, 0,
        {descriptorSet->uniform =
             vx::UniformBufferDescriptor(s->m_Buffer, 0, VK_WHOLE_SIZE)});
    m_CommandBuffer.cmdBindPipeline(vk::PipelineBindPoint::eGraphics,
                                    m_GradientPipeline);
    m_CommandBuffer.cmdPushConstants(m_GradientPipelineLayout,
                                     vk::ShaderStageFlagBits::bVertex,
                                     FIELD(VsInput, translate), &translation);
    g->Draw(m_CommandBuffer);
}

void Renderer_VX::Destroy(ShaderResource& s) {
    const auto allocator = m_Context->GetAllocator();
    allocator.destroyBuffer(s.m_Buffer, s.m_Allocation);
}

void Renderer_VX::ReleaseShader(Rml::CompiledShaderHandle shader) {
    RMLUI_ASSERT(shader);
    m_ShaderResources.Release(*this, ~shader);
}

void Renderer_VX::InitPipelineLayouts() {
    const auto device = m_Context->GetDevice();

    check(device.createTypedDescriptorSetLayout(
        &m_TextureDescriptorSetLayout,
        vk::DescriptorSetLayoutCreateFlagBits::bPushDescriptorKHR));

    check(device.createTypedDescriptorSetLayout(
        &m_GradientDescriptorSetLayout,
        vk::DescriptorSetLayoutCreateFlagBits::bPushDescriptorKHR));

    vk::PushConstantRange pushConstantRange;
    pushConstantRange.setStageFlags(vk::ShaderStageFlagBits::bVertex);
    pushConstantRange.setOffset(0);
    pushConstantRange.setSize(sizeof(VsInput));

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    pipelineLayoutInfo.setPushConstantRangeCount(1);
    pipelineLayoutInfo.setPushConstantRanges(&pushConstantRange);

    m_BasicPipelineLayout =
        device.createPipelineLayout(pipelineLayoutInfo).get();

    pipelineLayoutInfo.setSetLayoutCount(1);
    pipelineLayoutInfo.setSetLayouts(&m_TextureDescriptorSetLayout);

    m_TexturePipelineLayout =
        device.createPipelineLayout(pipelineLayoutInfo).get();

    pipelineLayoutInfo.setSetLayouts(&m_GradientDescriptorSetLayout);

    m_GradientPipelineLayout =
        device.createPipelineLayout(pipelineLayoutInfo).get();
}

void Renderer_VX::InitPipelines(
    vk::PipelineRenderingCreateInfo& renderingInfo) {
    const auto device = m_Context->GetDevice();

    const auto clipVertShader =
        device.createShaderModule(shader_vert_clip).get();
    const auto mainVertShader =
        device.createShaderModule(shader_vert_main).get();
    const auto colorFragShader =
        device.createShaderModule(shader_frag_color).get();
    const auto textureFragShader =
        device.createShaderModule(shader_frag_texture).get();
    const auto gradientFragShader =
        device.createShaderModule(shader_frag_gradient).get();

    vx::GraphicsPipelineBuilder pipelineBuilder;
    pipelineBuilder.attach(renderingInfo);
    pipelineBuilder.setTopology(vk::PrimitiveTopology::eTriangleList);
    pipelineBuilder.setPolygonMode(vk::PolygonMode::eFill);
    pipelineBuilder.setFrontFace(vk::FrontFace::eClockwise);
    pipelineBuilder.setCullMode(vk::CullModeFlagBits::eNone);

    vk::PipelineShaderStageCreateInfo shaderStageInfos[2];
    pipelineBuilder.setStages(shaderStageInfos);

    vk::DynamicState dynamicStates[4];
    pipelineBuilder.setDynamicStates(dynamicStates);

    vk::VertexInputAttributeDescription vertexAttributeDescriptions[3];
    pipelineBuilder.setVertexAttributeDescriptions(vertexAttributeDescriptions);

    shaderStageInfos[0] = vx::makePipelineShaderStageCreateInfo(
        vk::ShaderStageFlagBits::bVertex, {});
    pipelineBuilder.setStageCount(1);

    dynamicStates[0] = vk::DynamicState::eViewport;
    dynamicStates[1] = vk::DynamicState::eScissor;
    dynamicStates[2] = vk::DynamicState::eStencilOp;
    pipelineBuilder.setDynamicStateCount(3);

    pipelineBuilder.setStencilTestEnable(true);
    vk::StencilOpState stencilOp;
    stencilOp.setCompareMask(~0u);
    stencilOp.setReference(1);
    stencilOp.setWriteMask(~0u);
    pipelineBuilder.setStencilOpState(stencilOp);

    vk::VertexInputBindingDescription vertexBindingDescriptions[1];
    vertexBindingDescriptions[0].setBinding(0);
    vertexBindingDescriptions[0].setStride(sizeof(Rml::Vertex));
    vertexBindingDescriptions[0].setInputRate(vk::VertexInputRate::eVertex);
    pipelineBuilder.setVertexBindingDescriptionCount(1);
    pipelineBuilder.setVertexBindingDescriptions(vertexBindingDescriptions);

    vertexAttributeDescriptions[0].setLocation(0);
    vertexAttributeDescriptions[0].setBinding(0);
    vertexAttributeDescriptions[0].setFormat(vk::Format::eR32G32Sfloat);
    vertexAttributeDescriptions[0].setOffset(offsetof(Rml::Vertex, position));
    pipelineBuilder.setVertexAttributeDescriptionCount(1);

    pipelineBuilder.setLayout(m_BasicPipelineLayout);
    shaderStageInfos[0].setModule(clipVertShader);

    m_ClipPipeline = pipelineBuilder.build(device).get();

    shaderStageInfos[1] = vx::makePipelineShaderStageCreateInfo(
        vk::ShaderStageFlagBits::bFragment, {});
    pipelineBuilder.setStageCount(2);

    dynamicStates[2] = vk::DynamicState::eStencilTestEnable;
    dynamicStates[3] = vk::DynamicState::eStencilReference;
    pipelineBuilder.setDynamicStateCount(4);

    stencilOp.setCompareOp(vk::CompareOp::eEqual);
    stencilOp.setWriteMask(0);
    stencilOp.setFailOp(vk::StencilOp::eKeep);
    stencilOp.setPassOp(vk::StencilOp::eKeep);
    stencilOp.setDepthFailOp(vk::StencilOp::eKeep);
    pipelineBuilder.setStencilOpState(stencilOp);

    pipelineBuilder.setColorWriteMask(
        vk::ColorComponentFlagBits::bR | vk::ColorComponentFlagBits::bG |
        vk::ColorComponentFlagBits::bB | vk::ColorComponentFlagBits::bA);
    pipelineBuilder.setBlendEnable(true);
    pipelineBuilder.setBlend(vk::BlendOp::eAdd, vk::BlendFactor::eOne,
                             vk::BlendFactor::eOneMinusSrcAlpha);

    vertexAttributeDescriptions[1].setLocation(1);
    vertexAttributeDescriptions[1].setBinding(0);
    vertexAttributeDescriptions[1].setFormat(vk::Format::eR8G8B8A8Unorm);
    vertexAttributeDescriptions[1].setOffset(offsetof(Rml::Vertex, colour));
    vertexAttributeDescriptions[2].setLocation(2);
    vertexAttributeDescriptions[2].setBinding(0);
    vertexAttributeDescriptions[2].setFormat(vk::Format::eR32G32Sfloat);
    vertexAttributeDescriptions[2].setOffset(offsetof(Rml::Vertex, tex_coord));
    pipelineBuilder.setVertexAttributeDescriptionCount(3);

    shaderStageInfos[0].setModule(mainVertShader);
    shaderStageInfos[1].setModule(colorFragShader);

    m_ColorPipeline = pipelineBuilder.build(device).get();

    pipelineBuilder.setLayout(m_TexturePipelineLayout);
    shaderStageInfos[1].setModule(textureFragShader);

    m_TexturePipeline = pipelineBuilder.build(device).get();

    pipelineBuilder.setLayout(m_GradientPipelineLayout);
    shaderStageInfos[1].setModule(gradientFragShader);
    m_GradientPipeline = pipelineBuilder.build(device).get();

    device.destroyShaderModule(clipVertShader);
    device.destroyShaderModule(mainVertShader);
    device.destroyShaderModule(colorFragShader);
    device.destroyShaderModule(textureFragShader);
    device.destroyShaderModule(gradientFragShader);
}

Rml::TextureHandle Renderer_VX::CreateTexture(vk::Buffer buffer,
                                              Rml::Vector2i dimensions) {
    const auto device = m_Context->GetDevice();
    const auto allocator = m_Context->GetAllocator();

    const auto [index, t] = m_TextureResources.Allocate();

    const vk::Extent2D extent(dimensions.x, dimensions.y);
    const auto imageInfo =
        vx::image2DCreateInfo(vk::Format::eR8G8B8A8Unorm, extent,
                              vk::ImageUsageFlagBits::bSampled |
                                  vk::ImageUsageFlagBits::bTransferDst);

    vma::AllocationCreateInfo allocationInfo;
    allocationInfo.setUsage(vma::MemoryUsage::eAutoPreferDevice);

    t->m_Image =
        allocator.createImage(imageInfo, allocationInfo, &t->m_Allocation)
            .get();

    const auto commandBuffer = m_Context->BeginTemp();

    vx::ImageMemoryBarrierState imageMemoryBarrier(
        t->m_Image, vx::allSubresourceRange(vk::ImageAspectFlagBits::bColor));

    imageMemoryBarrier.update(vk::ImageLayout::eTransferDstOptimal,
                              vk::PipelineStageFlagBits2::bCopy,
                              vk::AccessFlagBits2::bTransferWrite);
    commandBuffer.cmdPipelineBarriers(imageMemoryBarrier);

    vk::BufferImageCopy bufferImageCopy;
    bufferImageCopy.setImageSubresource(
        vx::subresourceLayers(vk::ImageAspectFlagBits::bColor, 1));
    bufferImageCopy.setImageExtent(vx::toExtent3D(extent));
    commandBuffer.cmdCopyBufferToImage(buffer, t->m_Image,
                                       vk::ImageLayout::eTransferDstOptimal, 1,
                                       &bufferImageCopy);

    imageMemoryBarrier.update(vk::ImageLayout::eShaderReadOnlyOptimal,
                              vk::PipelineStageFlagBits2::bFragmentShader,
                              vk::AccessFlagBits2::bShaderRead);
    commandBuffer.cmdPipelineBarriers(imageMemoryBarrier);

    m_Context->EndTemp(commandBuffer);

    const auto imageViewInfo = vx::imageViewCreateInfo(
        vk::ImageViewType::e2D, t->m_Image, imageInfo.getFormat(),
        vk::ImageAspectFlagBits::bColor);
    t->m_ImageView = device.createImageView(imageViewInfo).get();

    return ~index;
}