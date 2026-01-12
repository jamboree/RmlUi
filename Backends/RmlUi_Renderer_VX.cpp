#include "RmlUi_Renderer_VX.h"
#include <RmlUi/Core/Core.h>
#include <RmlUi/Core/DecorationTypes.h>
#include <RmlUi/Core/FileInterface.h>
#include <RmlUi/Core/Log.h>
#include "RmlUi_VX/ShadersCompiledSPV.h"
#include "RmlUi_VX/BlurDefines.h"

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

static inline vk::ColorBlendEquationEXT
MakeColorBlendEquation(vk::BlendOp op, vk::BlendFactor srcFactor,
                       vk::BlendFactor dstFactor) {
    vk::ColorBlendEquationEXT colorBlendEquation;
    colorBlendEquation.setColorBlendOp(op);
    colorBlendEquation.setAlphaBlendOp(op);
    colorBlendEquation.setSrcColorBlendFactor(srcFactor);
    colorBlendEquation.setSrcAlphaBlendFactor(srcFactor);
    colorBlendEquation.setDstColorBlendFactor(dstFactor);
    colorBlendEquation.setDstAlphaBlendFactor(dstFactor);
    return colorBlendEquation;
}

static inline void SetViewport(vx::CommandBuffer commandBuffer, unsigned width,
                               unsigned height) {
    vk::Viewport viewport;
    viewport.setWidth(width);
    viewport.setHeight(height);
    viewport.setMinDepth(0.f);
    viewport.setMaxDepth(1.f);
    commandBuffer.cmdSetViewport(0, 1, &viewport);
}

struct QuadMesh {
    Rml::Vertex vertices[4];
    int indices[6];

    QuadMesh(Rml::Vector2f origin, Rml::Vector2f dimensions,
             Rml::Vector2f top_left_texcoord,
             Rml::Vector2f bottom_right_texcoord) {
        vertices[0].position = origin;
        vertices[0].tex_coord = top_left_texcoord;

        vertices[1].position = Rml::Vector2f(origin.x + dimensions.x, origin.y);
        vertices[1].tex_coord =
            Rml::Vector2f(bottom_right_texcoord.x, top_left_texcoord.y);

        vertices[2].position = origin + dimensions;
        vertices[2].tex_coord = bottom_right_texcoord;

        vertices[3].position = Rml::Vector2f(origin.x, origin.y + dimensions.y);
        vertices[3].tex_coord =
            Rml::Vector2f(top_left_texcoord.x, bottom_right_texcoord.y);

        indices[0] = 0;
        indices[1] = 3;
        indices[2] = 1;

        indices[3] = 1;
        indices[4] = 3;
        indices[5] = 2;
    }
};

struct VsInput {
    Rml::Matrix4f transform;
    Rml::Vector2f translate;
};

struct ColorMatrixFsInput {
    uint8_t _pad[64];
    Rml::Matrix4f colorMatrix;
};

struct BlurVsInput {
    Rml::Matrix4f _pad;
    Rml::Vector2f texelOffset;
};

struct BlurFsInput {
    Rml::Vector2f texCoordMin;
    Rml::Vector2f texCoordMax;
    float weights[BLUR_NUM_WEIGHTS];

    void SetTexCoordLimits(const vk::Rect2D& rectangle_flipped,
                           vk::Extent2D framebuffer_size) {
        // Offset by half-texel values so that texture lookups are clamped to
        // fragment centers, thereby avoiding color bleeding from neighboring
        // texels due to bilinear interpolation.
        texCoordMin.x =
            (rectangle_flipped.offset.x + 0.5f) / framebuffer_size.width;
        texCoordMin.y =
            (rectangle_flipped.offset.y + 0.5f) / framebuffer_size.height;

        texCoordMax.x = (rectangle_flipped.offset.x +
                         rectangle_flipped.extent.width - 0.5f) /
                        framebuffer_size.width;
        texCoordMax.y = (rectangle_flipped.offset.y +
                         rectangle_flipped.extent.height - 0.5f) /
                        framebuffer_size.height;
    }

    void SetBlurWeights(float sigma) {
        constexpr int num_weights = BLUR_NUM_WEIGHTS;
        float normalization = 0.0f;
        for (int i = 0; i < num_weights; ++i) {
            if (Rml::Math::Absolute(sigma) < 0.1f) {
                weights[i] = float(i == 0);
            } else {
                weights[i] =
                    Rml::Math::Exp(-float(i * i) / (2.0f * sigma * sigma)) /
                    (Rml::Math::SquareRoot(2.f * Rml::Math::RMLUI_PI) * sigma);
            }
            normalization += (i == 0 ? 1.f : 2.0f) * weights[i];
        }
        for (int i = 0; i < num_weights; ++i)
            weights[i] /= normalization;
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

struct Renderer_VX::TextureDescriptorSet {
    static constexpr vk::ShaderStageFlags Stages =
        vk::ShaderStageFlagBits::bFragment;

    VX_BINDING(0, vx::CombinedImageImmutableSamplerDescriptor<0>, Stages) tex;
};

struct Renderer_VX::UniformDescriptorSet {
    static constexpr vk::ShaderStageFlags Stages =
        vk::ShaderStageFlagBits::bFragment;

    VX_BINDING(0, vx::UniformBufferDescriptor, Stages) uniform;
};

struct Renderer_VX::BlurDescriptorSet {
    static constexpr vk::ShaderStageFlags Stages =
        vk::ShaderStageFlagBits::bFragment;

    VX_BINDING(0, vx::CombinedImageImmutableSamplerDescriptor<0>, Stages) tex;
    VX_BINDING(1, vx::UniformBufferDescriptor, Stages) input;
};

struct Renderer_VX::FilterDescriptorSet {
    static constexpr vk::ShaderStageFlags Stages =
        vk::ShaderStageFlagBits::bFragment;
    static constexpr auto BindingFlags =
        vk::DescriptorBindingFlagBits::bUpdateAfterBind |
        vk::DescriptorBindingFlagBits::bPartiallyBound;

    VX_BINDING(0, vx::ImmutableSamplerDescriptor<0>, Stages) mySampler;
    VX_BINDING(1, vx::SampledImageDescriptor[4], Stages, BindingFlags) textures;
};

struct Renderer_VX::LayerState {
    enum State : uint8_t { None, Transfer, Attachment };
    State m_State = None;

    void SetImageBarrierSrc(vx::ImageMemoryBarrierState& imageBarrier) const {
        switch (m_State) {
        case Transfer:
            imageBarrier.setOldLayout(vk::ImageLayout::eTransferSrcOptimal);
            imageBarrier.setSrcStageAccess(
                vk::PipelineStageFlagBits2::bTransfer,
                vk::AccessFlagBits2::bTransferRead);
            break;
        case Attachment:
            imageBarrier.setOldLayout(vk::ImageLayout::eAttachmentOptimal);
            imageBarrier.setSrcStageAccess(
                vk::PipelineStageFlagBits2::bColorAttachmentOutput,
                vk::AccessFlagBits2::bColorAttachmentWrite);
            break;
        }
    }
};

Renderer_VX::SurfaceManager::~SurfaceManager() {
    std::free(m_layers);
    std::free(m_layerStates);
}

void Renderer_VX::SurfaceManager::Destroy(GfxContext_VX& gfx) {
    for (auto& image : std::span(m_layers, m_layers_capacity)) {
        gfx.DestroyImageAttachment(image);
    }
    for (auto& image : m_postprocess) {
        gfx.DestroyImageAttachment(image);
    }
}

void Renderer_VX::SurfaceManager::Invalidate() {
    m_layers_capacity = 0;
    for (auto& image : m_postprocess) {
        image.m_Image = {};
    }
}

Rml::LayerHandle Renderer_VX::SurfaceManager::PushLayer(GfxContext_VX& gfx) {
    RMLUI_ASSERT(m_layers_size <= m_layers_capacity);

    if (m_layers_size == m_layers_capacity) {
        ++m_layers_capacity;
        m_layers = static_cast<ImageAttachment*>(std::realloc(
            m_layers, sizeof(ImageAttachment) * m_layers_capacity));
        m_layers[m_layers_size] = gfx.CreateImageAttachment(
            gfx.m_SwapchainImageFormat,
            vk::ImageUsageFlagBits::bColorAttachment |
                vk::ImageUsageFlagBits::bTransferSrc,
            vk::ImageAspectFlagBits::bColor, gfx.m_SampleCount);
        m_layerStates = static_cast<LayerState*>(std::realloc(
            m_layerStates, sizeof(LayerState) * m_layers_capacity));
    }
    m_layerStates[m_layers_size] = {};

    ++m_layers_size;
    return GetTopLayerHandle();
}

Renderer_VX::LayerState&
Renderer_VX::SurfaceManager::GetLayerState(Rml::LayerHandle layer) {
    RMLUI_ASSERT((size_t)layer < (size_t)m_layers_size);
    return m_layerStates[layer];
}

Renderer_VX::Renderer_VX() = default;
Renderer_VX::~Renderer_VX() = default;

bool Renderer_VX::Init(GfxContext_VX& gfx) {
    m_Gfx = &gfx;

    const auto device = m_Gfx->m_Device;

    vk::SamplerCreateInfo samplerInfo;
    samplerInfo.setMagFilter(vk::Filter::eLinear);
    samplerInfo.setMinFilter(vk::Filter::eLinear);
    samplerInfo.setAddressModeU(vk::SamplerAddressMode::eRepeat);
    samplerInfo.setAddressModeV(vk::SamplerAddressMode::eRepeat);
    samplerInfo.setAddressModeW(vk::SamplerAddressMode::eRepeat);
    m_Sampler = device.createSampler(samplerInfo).get();

    vk::DescriptorPoolCreateInfo descriptorPoolInfo;
    descriptorPoolInfo.setFlags(
        vk::DescriptorPoolCreateFlagBits::bUpdateAfterBind);
    descriptorPoolInfo.setMaxSets(100);
    vk::DescriptorPoolSize poolSizes[] = {
        {vk::DescriptorType::eSampler, 1},
        {vk::DescriptorType::eSampledImage, 4}};
    descriptorPoolInfo.setPoolSizeCount(std::size(poolSizes));
    m_DescriptorPool = device.createDescriptorPool(descriptorPoolInfo).get();

    m_FilterDescriptorSet =
        device
            .allocateTypedDescriptorSet<FilterDescriptorSet>(
                m_DescriptorPool, m_FilterDescriptorSetLayout)
            .get();
    device.updateDescriptorSets({
        m_FilterDescriptorSet->mySampler = vk::Sampler(), // immutable
    });

    InitPipelineLayouts();

    vk::PipelineRenderingCreateInfo renderingInfo;
    renderingInfo.setColorAttachmentCount(1);
    renderingInfo.setColorAttachmentFormats(&gfx.m_SwapchainImageFormat);
    renderingInfo.setDepthAttachmentFormat(gfx.m_DepthStencilImageFormat);
    renderingInfo.setStencilAttachmentFormat(gfx.m_DepthStencilImageFormat);

    InitPipelines(renderingInfo);

    const QuadMesh mesh(Rml::Vector2f(-1), Rml::Vector2f(2),
                        Rml::Vector2f(0, 0), Rml::Vector2f(1, 1));
    m_FullscreenQuadGeometry = CompileGeometry(mesh.vertices, mesh.indices);

    return true;
}

void Renderer_VX::Destroy() {
    const auto device = m_Gfx->m_Device;

    m_SurfaceManager.Destroy(*m_Gfx);
    if (m_FullscreenQuadGeometry) {
        ReleaseGeometry(m_FullscreenQuadGeometry);
    }

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
    if (m_PassthroughPipeline) {
        device.destroyPipeline(m_PassthroughPipeline);
    }
    if (m_MsPassthroughPipeline) {
        device.destroyPipeline(m_MsPassthroughPipeline);
    }
    if (m_ColorMatrixPipeline) {
        device.destroyPipeline(m_ColorMatrixPipeline);
    }
    if (m_BlendMaskPipeline) {
        device.destroyPipeline(m_BlendMaskPipeline);
    }
    if (m_BlurPipeline) {
        device.destroyPipeline(m_BlurPipeline);
    }
    if (m_BasicPipelineLayout) {
        device.destroyPipelineLayout(m_BasicPipelineLayout);
    }
    if (m_GradientPipelineLayout) {
        device.destroyPipelineLayout(m_GradientPipelineLayout);
    }
    if (m_TexturePipelineLayout) {
        device.destroyPipelineLayout(m_TexturePipelineLayout);
    }
    if (m_FilterPipelineLayout) {
        device.destroyPipelineLayout(m_FilterPipelineLayout);
    }
    if (m_BlurPipelineLayout) {
        device.destroyPipelineLayout(m_BlurPipelineLayout);
    }
    if (m_TextureDescriptorSetLayout) {
        device.destroyDescriptorSetLayout(m_TextureDescriptorSetLayout);
    }
    if (m_UniformDescriptorSetLayout) {
        device.destroyDescriptorSetLayout(m_UniformDescriptorSetLayout);
    }
    if (m_BlurDescriptorSetLayout) {
        device.destroyDescriptorSetLayout(m_BlurDescriptorSetLayout);
    }
    if (m_FilterDescriptorSetLayout) {
        device.destroyDescriptorSetLayout(m_FilterDescriptorSetLayout);
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

void Renderer_VX::BeginFrame(vx::CommandBuffer commandBuffer) {
    m_CommandBuffer = commandBuffer;
    m_StencilRef = 1;
    m_PostprocessPrimaryIndex = 0;
    m_EnableScissor = false;
    m_EnableClipMask = false;
    m_LayerRendering = false;
    m_DepthStencilInitialized = false;

    const auto extent = m_Gfx->m_FrameExtent;

    SetViewport(m_CommandBuffer, extent.width, extent.height);

    m_Scissor.setOffset({});
    m_Scissor.setExtent(extent);
    m_CommandBuffer.cmdSetScissor(0, 1, &m_Scissor);

    const auto transform = Project(extent, Rml::Matrix4f::Identity());
    m_CommandBuffer.cmdPushConstant(m_BasicPipelineLayout,
                                    vk::ShaderStageFlagBits::bVertex,
                                    VX_FIELD(VsInput, transform) = transform);
    m_CommandBuffer.cmdSetStencilTestEnable(m_EnableClipMask);
    m_CommandBuffer.cmdSetStencilReference(
        vk::StencilFaceFlagBits::eFrontAndBack, m_StencilRef);

    {
        vk::BindDescriptorSetsInfo bindDescriptorSetsInfo;
        bindDescriptorSetsInfo.setLayout(m_TexturePipelineLayout);
        bindDescriptorSetsInfo.setFirstSet(0);
        bindDescriptorSetsInfo.setDescriptorSetCount(1);
        bindDescriptorSetsInfo.setDescriptorSets(&m_FilterDescriptorSet);
        bindDescriptorSetsInfo.setStageFlags(
            vk::ShaderStageFlagBits::bFragment);
        m_CommandBuffer.cmdBindDescriptorSets2(bindDescriptorSetsInfo);
    }

    const auto topLayer = m_SurfaceManager.PushLayer(*m_Gfx);
    BeginLayerRendering(topLayer);
}

void Renderer_VX::EndFrame() {
    RMLUI_ASSERT(m_SurfaceManager.GetTopLayerHandle() == 0);
    EndLayerRendering();
    ResolveLayer(0, m_Gfx->CurrentFrameResource().m_Image);
    m_SurfaceManager.PopLayer();
}

void Renderer_VX::ResetRenderTarget() {
    m_SurfaceManager.Destroy(*m_Gfx);
    m_SurfaceManager.Invalidate();
}

void Renderer_VX::ReleaseAllResourceUse(uint8_t useFlags) {
    m_GeometryResources.ReleaseAllUse(*this, useFlags);
    m_TextureResources.ReleaseAllUse(*this, useFlags);
    m_ShaderResources.ReleaseAllUse(*this, useFlags);
}

Rml::CompiledGeometryHandle
Renderer_VX::CompileGeometry(Rml::Span<const Rml::Vertex> vertices,
                             Rml::Span<const int> indices) {
    return ~m_GeometryResources.Create(CreateGeometry(vertices, indices));
}

void Renderer_VX::RenderGeometry(Rml::CompiledGeometryHandle geometry,
                                 Rml::Vector2f translation,
                                 Rml::TextureHandle texture) {
    const uint8_t useFlag = 2u << m_Gfx->m_FrameNumber;
    const auto& g = m_GeometryResources.Use(~geometry, useFlag);
    auto pipeline = m_ColorPipeline;
    auto pipelineLayout = m_BasicPipelineLayout;

    ActivateLayerRendering();

    if (texture) {
        pipeline = m_TexturePipeline;
        pipelineLayout = m_TexturePipelineLayout;
        const auto& t = m_TextureResources.Use(~texture, useFlag);
        const vx::DescriptorSet<TextureDescriptorSet> descriptorSet;
        m_CommandBuffer.cmdPushDescriptorSetKHR(
            vk::PipelineBindPoint::eGraphics, pipelineLayout, 0,
            {descriptorSet->tex = t.m_ImageView});
    }
    m_CommandBuffer.cmdBindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

    m_CommandBuffer.cmdPushConstant(pipelineLayout,
                                    vk::ShaderStageFlagBits::bVertex,
                                    VX_FIELD(VsInput, translate) = translation);
    g.Draw(m_CommandBuffer);
}

void Renderer_VX::DestroyResource(GeometryResource& g) {
    const auto allocator = m_Gfx->m_Allocator;
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
    StagingBuffer stagingBuffer{m_Gfx->m_Allocator};

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
    StagingBuffer stagingBuffer{m_Gfx->m_Allocator};
    std::memcpy(stagingBuffer.Alloc(source_data.size()), source_data.data(),
                source_data.size());
    return CreateTexture(stagingBuffer.m_Buffer, source_dimensions);
}

void Renderer_VX::DestroyResource(TextureResource& t) {
    const auto device = m_Gfx->m_Device;
    const auto allocator = m_Gfx->m_Allocator;
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
        m_Scissor.setExtent(m_Gfx->m_FrameExtent);
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
    const auto extent = m_Gfx->m_FrameExtent;
    const auto matrix =
        Project(extent, transform ? *transform : Rml::Matrix4f::Identity());
    m_CommandBuffer.cmdPushConstant(m_BasicPipelineLayout,
                                    vk::ShaderStageFlagBits::bVertex,
                                    VX_FIELD(VsInput, transform) = matrix);
}

void Renderer_VX::EnableClipMask(bool enable) {
    m_EnableClipMask = enable;
    m_CommandBuffer.cmdSetStencilTestEnable(m_EnableClipMask);
}

void Renderer_VX::RenderToClipMask(Rml::ClipMaskOperation operation,
                                   Rml::CompiledGeometryHandle geometry,
                                   Rml::Vector2f translation) {
    const uint8_t useFlag = 2u << m_Gfx->m_FrameNumber;
    const auto& g = m_GeometryResources.Use(~geometry, useFlag);

    bool clearStencil = false;
    auto stencilPassOp = vk::StencilOp::eReplace;
    uint32_t stencilWriteValue = 1;
    uint32_t stencilTestValue = 1;
    uint32_t stencilClearValue = 1;
    switch (operation) {
    case Rml::ClipMaskOperation::Set:
        clearStencil = true;
        stencilClearValue = 0;
        break;
    case Rml::ClipMaskOperation::SetInverse:
        clearStencil = true;
        stencilWriteValue = 0;
        break;
    case Rml::ClipMaskOperation::Intersect:
        stencilPassOp = vk::StencilOp::eIncrementAndClamp;
        stencilTestValue = m_StencilRef + 1;
        break;
    }

    ActivateLayerRendering();

    if (clearStencil && m_Scissor.extent.width && m_Scissor.extent.height) {
        vk::ClearAttachment clearAttachment;
        clearAttachment.setAspectMask(vk::ImageAspectFlagBits::bStencil);
        clearAttachment.setClearValue(
            {.depthStencil =
                 vk::ClearDepthStencilValue{1.0f, stencilClearValue}});
        vk::ClearRect clearRect;
        clearRect.setLayerCount(1);
        clearRect.setRect(m_Scissor);
        m_CommandBuffer.cmdClearAttachments(1, &clearAttachment, 1, &clearRect);
    }

    m_CommandBuffer.cmdBindPipeline(vk::PipelineBindPoint::eGraphics,
                                    m_ClipPipeline);
    m_CommandBuffer.cmdPushConstant(m_BasicPipelineLayout,
                                    vk::ShaderStageFlagBits::bVertex,
                                    VX_FIELD(VsInput, translate) = translation);
    m_CommandBuffer.cmdSetStencilOp(
        vk::StencilFaceFlagBits::eFrontAndBack, vk::StencilOp::eKeep,
        stencilPassOp, vk::StencilOp::eKeep, vk::CompareOp::eAlways);
    m_CommandBuffer.cmdSetStencilReference(
        vk::StencilFaceFlagBits::eFrontAndBack, stencilWriteValue);

    g.Draw(m_CommandBuffer);

    m_CommandBuffer.cmdSetStencilTestEnable(m_EnableClipMask);
    m_CommandBuffer.cmdSetStencilReference(
        vk::StencilFaceFlagBits::eFrontAndBack, stencilTestValue);
    m_StencilRef = stencilTestValue;
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

    return ~m_ShaderResources.Create(
        CreateShaderResource(&uniform, uniform.GetUsedSize()));
}

void Renderer_VX::RenderShader(Rml::CompiledShaderHandle shader,
                               Rml::CompiledGeometryHandle geometry,
                               Rml::Vector2f translation,
                               Rml::TextureHandle /*texture*/) {
    const uint8_t useFlag = 2u << m_Gfx->m_FrameNumber;
    const auto& s = m_ShaderResources.Use(~shader, useFlag);
    const auto& g = m_GeometryResources.Use(~geometry, useFlag);

    ActivateLayerRendering();

    const vx::DescriptorSet<UniformDescriptorSet> descriptorSet;
    m_CommandBuffer.cmdPushDescriptorSetKHR(
        vk::PipelineBindPoint::eGraphics, m_GradientPipelineLayout, 0,
        {descriptorSet->uniform = vx::UniformBufferDescriptor(s.m_Buffer)});
    m_CommandBuffer.cmdBindPipeline(vk::PipelineBindPoint::eGraphics,
                                    m_GradientPipeline);

    m_CommandBuffer.cmdPushConstant(m_GradientPipelineLayout,
                                    vk::ShaderStageFlagBits::bVertex,
                                    VX_FIELD(VsInput, translate) = translation);
    g.Draw(m_CommandBuffer);
}

void Renderer_VX::DestroyResource(ShaderResource& s) {
    const auto allocator = m_Gfx->m_Allocator;
    allocator.destroyBuffer(s.m_Buffer, s.m_Allocation);
}

void Renderer_VX::BeginLayerRendering(Rml::LayerHandle handle) {
    RMLUI_ASSERT(!m_LayerRendering);
    m_CurrentLayer = handle;
}

void Renderer_VX::ActivateLayerRendering() {
    if (m_LayerRendering) {
        return;
    }
    m_LayerRendering = true;

    const auto& colorImage = m_SurfaceManager.GetLayer(m_CurrentLayer);
    auto& layerState = m_SurfaceManager.GetLayerState(m_CurrentLayer);
    vx::ImageMemoryBarrierState imageMemoryBarriers[2];
    auto& [colorImageBarrier, depthStencilImageBarrier] = imageMemoryBarriers;

    colorImageBarrier.init(
        colorImage.m_Image,
        vx::subresourceRange(vk::ImageAspectFlagBits::bColor));
    layerState.SetImageBarrierSrc(colorImageBarrier);
    colorImageBarrier.setNewLayout(vk::ImageLayout::eAttachmentOptimal);
    colorImageBarrier.setDstStageAccess(
        vk::PipelineStageFlagBits2::bColorAttachmentOutput,
        vk::AccessFlagBits2::bColorAttachmentWrite);

    depthStencilImageBarrier.init(
        m_Gfx->m_DepthStencilImage.m_Image,
        vx::subresourceRange(vk::ImageAspectFlagBits::bDepth |
                             vk::ImageAspectFlagBits::bStencil));
    if (m_DepthStencilInitialized) {
        depthStencilImageBarrier.setOldLayout(
            vk::ImageLayout::eAttachmentOptimal);
    }
    depthStencilImageBarrier.setSrcStageAccess(
        vk::PipelineStageFlagBits2::bLateFragmentTests,
        vk::AccessFlagBits2::bDepthStencilAttachmentWrite);
    depthStencilImageBarrier.setNewLayout(vk::ImageLayout::eAttachmentOptimal);
    depthStencilImageBarrier.setDstStageAccess(
        vk::PipelineStageFlagBits2::bEarlyFragmentTests |
            vk::PipelineStageFlagBits2::bLateFragmentTests,
        vk::AccessFlagBits2::bDepthStencilAttachmentRead |
            vk::AccessFlagBits2::bDepthStencilAttachmentWrite);

    m_CommandBuffer.cmdPipelineBarriers(rawIns(imageMemoryBarriers));

    vk::RenderingAttachmentInfo colorAttachmentInfo;
    colorAttachmentInfo.setImageView(colorImage.m_ImageView);
    colorAttachmentInfo.setImageLayout(colorImageBarrier.getNewLayout());
    colorAttachmentInfo.setStoreOp(vk::AttachmentStoreOp::eStore);
    if (layerState.m_State == LayerState::None) {
        colorAttachmentInfo.setLoadOp(vk::AttachmentLoadOp::eClear);
        colorAttachmentInfo.setClearValue({.color = vk::ClearColorValue()});
    } else {
        colorAttachmentInfo.setLoadOp(vk::AttachmentLoadOp::eLoad);
    }

    vk::RenderingAttachmentInfo depthStencilAttachmentInfo;
    depthStencilAttachmentInfo.setImageLayout(
        depthStencilImageBarrier.getNewLayout());
    depthStencilAttachmentInfo.setStoreOp(vk::AttachmentStoreOp::eStore);
    depthStencilAttachmentInfo.setImageView(
        m_Gfx->m_DepthStencilImage.m_ImageView);
    if (m_DepthStencilInitialized) {
        depthStencilAttachmentInfo.setLoadOp(vk::AttachmentLoadOp::eLoad);
    } else {
        depthStencilAttachmentInfo.setLoadOp(vk::AttachmentLoadOp::eClear);
        depthStencilAttachmentInfo.setClearValue(
            {.depthStencil = vk::ClearDepthStencilValue{1.0f, 0}});
    }

    layerState.m_State = LayerState::Attachment;
    m_DepthStencilInitialized = true;

    const vk::Rect2D renderArea{{0, 0}, m_Gfx->m_FrameExtent};

    vk::RenderingInfo renderingInfo;
    renderingInfo.setRenderArea(renderArea);
    renderingInfo.setLayerCount(1);
    renderingInfo.setColorAttachmentCount(1);
    renderingInfo.setColorAttachments(&colorAttachmentInfo);
    renderingInfo.setDepthAttachment(&depthStencilAttachmentInfo);
    renderingInfo.setStencilAttachment(&depthStencilAttachmentInfo);

    m_CommandBuffer.cmdBeginRendering(renderingInfo);
}

void Renderer_VX::EndLayerRendering() {
    if (m_LayerRendering) {
        m_CommandBuffer.cmdEndRendering();
        m_LayerRendering = false;
    }
}

const ImageAttachment& Renderer_VX::GetPostprocess(unsigned index) {
    RMLUI_ASSERT(index < std::size(m_SurfaceManager.m_postprocess));
    auto& fb = m_SurfaceManager.m_postprocess[index];
    if (!fb.m_Image) {
        fb = m_Gfx->CreateImageAttachment(
            m_Gfx->m_SwapchainImageFormat,
            vk::ImageUsageFlagBits::bColorAttachment |
                vk::ImageUsageFlagBits::bSampled |
                vk::ImageUsageFlagBits::bTransferSrc |
                vk::ImageUsageFlagBits::bTransferDst,
            vk::ImageAspectFlagBits::bColor, vk::SampleCountFlagBits::b1);
        m_Gfx->m_Device.updateDescriptorSets(
            {m_FilterDescriptorSet->textures[index] = fb.m_ImageView});
    }
    return fb;
}

vk::Image Renderer_VX::BeginPostprocess(unsigned index) {
    const auto& colorImage = GetPostprocess(index);
    vx::ImageMemoryBarrierState colorImageBarrier;

    colorImageBarrier.init(
        colorImage.m_Image,
        vx::subresourceRange(vk::ImageAspectFlagBits::bColor));
    colorImageBarrier.setNewLayout(vk::ImageLayout::eAttachmentOptimal);
    colorImageBarrier.setSrcStageAccess(
        vk::PipelineStageFlagBits2::bFragmentShader,
        vk::AccessFlagBits2::bShaderRead);
    colorImageBarrier.setDstStageAccess(
        vk::PipelineStageFlagBits2::bColorAttachmentOutput,
        vk::AccessFlagBits2::bColorAttachmentWrite);

    m_CommandBuffer.cmdPipelineBarriers(colorImageBarrier);

    vk::RenderingAttachmentInfo colorAttachmentInfo;
    colorAttachmentInfo.setImageView(colorImage.m_ImageView);
    colorAttachmentInfo.setImageLayout(colorImageBarrier.getNewLayout());
    colorAttachmentInfo.setLoadOp(vk::AttachmentLoadOp::eClear);
    colorAttachmentInfo.setStoreOp(vk::AttachmentStoreOp::eStore);
    colorAttachmentInfo.setClearValue({.color = vk::ClearColorValue()});

    const vk::Rect2D renderArea{{0, 0}, m_Gfx->m_FrameExtent};

    vk::RenderingInfo renderingInfo;
    renderingInfo.setRenderArea(renderArea);
    renderingInfo.setLayerCount(1);
    renderingInfo.setColorAttachmentCount(1);
    renderingInfo.setColorAttachments(&colorAttachmentInfo);

    m_CommandBuffer.cmdBeginRendering(renderingInfo);

    return colorImage.m_Image;
}

void Renderer_VX::TransitionToSample(vk::Image image, bool fromTransfer) {
    vx::ImageMemoryBarrierState imageBarrier;
    imageBarrier.init(image,
                      vx::subresourceRange(vk::ImageAspectFlagBits::bColor));
    if (fromTransfer) {
        imageBarrier.setOldLayout(vk::ImageLayout::eTransferDstOptimal);
        imageBarrier.setSrcStageAccess(vk::PipelineStageFlagBits2::bTransfer,
                                       vk::AccessFlagBits2::bTransferWrite);
    } else {
        imageBarrier.setOldLayout(vk::ImageLayout::eAttachmentOptimal);
        imageBarrier.setSrcStageAccess(
            vk::PipelineStageFlagBits2::bColorAttachmentOutput,
            vk::AccessFlagBits2::bColorAttachmentWrite);
    }
    imageBarrier.setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
    imageBarrier.setDstStageAccess(vk::PipelineStageFlagBits2::bFragmentShader,
                                   vk::AccessFlagBits2::bShaderRead);

    m_CommandBuffer.cmdPipelineBarriers(imageBarrier);
}

struct TexInput {
    uint8_t _pad[128];
    unsigned texIdx;
};

void Renderer_VX::SetSample(unsigned index) {
    m_CommandBuffer.cmdPushConstant(m_FilterPipelineLayout,
                                    vk::ShaderStageFlagBits::bFragment,
                                    VX_FIELD(TexInput, texIdx) = index);
}

void Renderer_VX::ResolveLayer(Rml::LayerHandle source, vk::Image dstImage) {
    auto& layerState = m_SurfaceManager.GetLayerState(source);
    RMLUI_ASSERT(layerState.m_State != LayerState::None);

    vx::ImageMemoryBarrierState imageBarriers[2];
    auto& [srcImageBarrier, dstImageBarrier] = imageBarriers;

    srcImageBarrier.init(m_SurfaceManager.GetLayer(source).m_Image,
                         vx::subresourceRange(vk::ImageAspectFlagBits::bColor));
    layerState.SetImageBarrierSrc(srcImageBarrier);
    layerState.m_State = LayerState::Transfer;
    srcImageBarrier.setNewLayout(vk::ImageLayout::eTransferSrcOptimal);

    srcImageBarrier.setDstStageAccess(vk::PipelineStageFlagBits2::bTransfer,
                                      vk::AccessFlagBits2::bTransferRead);

    dstImageBarrier.init(dstImage,
                         vx::subresourceRange(vk::ImageAspectFlagBits::bColor));
    dstImageBarrier.setNewLayout(vk::ImageLayout::eTransferDstOptimal);
    dstImageBarrier.setDstStageAccess(vk::PipelineStageFlagBits2::bTransfer,
                                      vk::AccessFlagBits2::bTransferWrite);
    m_CommandBuffer.cmdPipelineBarriers(rawIns(imageBarriers));

    const auto setImageInfo = [&](auto& imageInfo, auto& imageRegion) {
        const auto subresource =
            vx::subresourceLayers(vk::ImageAspectFlagBits::bColor, 1);

        imageInfo.setSrcImage(srcImageBarrier.getImage());
        imageInfo.setDstImage(dstImageBarrier.getImage());
        imageInfo.setSrcImageLayout(srcImageBarrier.getNewLayout());
        imageInfo.setDstImageLayout(dstImageBarrier.getNewLayout());
        imageInfo.setRegionCount(1);
        imageRegion.setSrcSubresource(subresource);
        imageRegion.setDstSubresource(subresource);
        imageRegion.setExtent(
            {m_Gfx->m_FrameExtent.width, m_Gfx->m_FrameExtent.height, 1});
        imageInfo.setRegions(&imageRegion);
    };
    if (m_Gfx->m_SampleCount == vk::SampleCountFlagBits::b1) {
        vk::CopyImageInfo2 imageInfo;
        vk::ImageCopy2 imageRegion;
        setImageInfo(imageInfo, imageRegion);
        m_CommandBuffer.cmdCopyImage2(imageInfo);
    } else {
        vk::ResolveImageInfo2 imageInfo;
        vk::ImageResolve2 imageRegion;
        setImageInfo(imageInfo, imageRegion);
        m_CommandBuffer.cmdResolveImage2(imageInfo);
    }
}

Rml::CompiledGeometryHandle
Renderer_VX::UseFullscreenQuad(Rml::Vector2f uv_offset,
                               Rml::Vector2f uv_scaling) {
    if (uv_offset == Rml::Vector2f() && uv_scaling == Rml::Vector2f(1.f)) {
        return m_FullscreenQuadGeometry;
    }
    const QuadMesh mesh(Rml::Vector2f(-1), Rml::Vector2f(2), uv_offset,
                        uv_scaling);

    const uint8_t useFlag = 2u << m_Gfx->m_FrameNumber;
    return ~m_GeometryResources.Create(
        CreateGeometry(mesh.vertices, mesh.indices), useFlag);
}

void Renderer_VX::ReleaseShader(Rml::CompiledShaderHandle shader) {
    RMLUI_ASSERT(shader);
    m_ShaderResources.Release(*this, ~shader);
}

Rml::LayerHandle Renderer_VX::PushLayer() {
    EndLayerRendering();

    const auto topLayer = m_SurfaceManager.PushLayer(*m_Gfx);
    BeginLayerRendering(topLayer);
    return topLayer;
}

void Renderer_VX::CompositeLayers(
    Rml::LayerHandle source, Rml::LayerHandle destination,
    Rml::BlendMode blend_mode,
    Rml::Span<const Rml::CompiledFilterHandle> filters) {
    using Rml::BlendMode;

    if (!m_Scissor.extent.width || !m_Scissor.extent.height) {
        return;
    }

    EndLayerRendering();

    // Blit source layer to postprocessing buffer. Do this regardless of whether
    // we actually have any filters to be applied, because we need to resolve
    // the multi-sampled framebuffer in any case.
    // @performance If we have BlendMode::Replace and no filters or mask then we
    // can just blit directly to the destination.
    const auto postprocessImage = GetPostprocess(PostprocessPrimary()).m_Image;
    ResolveLayer(source, postprocessImage);

    TransitionToSample(postprocessImage, true);

    // Render the filters, the Postprocess(0) framebuffer is used for both
    // input and output.
    RenderFilters(filters);

    BeginLayerRendering(destination);
    ActivateLayerRendering();
    SetSample(PostprocessPrimary());

    m_CommandBuffer.cmdBindPipeline(vk::PipelineBindPoint::eGraphics,
                                    m_Gfx->m_SampleCount ==
                                            vk::SampleCountFlagBits::b1
                                        ? m_PassthroughPipeline
                                        : m_MsPassthroughPipeline);

    const vk::Bool32 colorBlendEnable = blend_mode == BlendMode::Blend;
    m_CommandBuffer.cmdSetColorBlendEnableEXT(0, 1, &colorBlendEnable);
    if (colorBlendEnable) {
        const auto colorBlendEquation =
            MakeColorBlendEquation(vk::BlendOp::eAdd, vk::BlendFactor::eOne,
                                   vk::BlendFactor::eOneMinusSrcAlpha);
        m_CommandBuffer.cmdSetColorBlendEquationEXT(0, 1, &colorBlendEquation);
    }

    m_GeometryResources.Get(~m_FullscreenQuadGeometry).Draw(m_CommandBuffer);

    const auto topLayer = m_SurfaceManager.GetTopLayerHandle();
    if (topLayer != destination) {
        EndLayerRendering();
        BeginLayerRendering(topLayer);
    }
}

void Renderer_VX::PopLayer() {
    EndLayerRendering();
    m_SurfaceManager.PopLayer();
    BeginLayerRendering(m_SurfaceManager.GetTopLayerHandle());
}

Rml::TextureHandle Renderer_VX::SaveLayerAsTexture() {
    TextureResource t;
    {
        const auto imageInfo = vx::image2DCreateInfo(
            m_Gfx->m_SwapchainImageFormat, m_Scissor.getExtent(),
            vk::ImageUsageFlagBits::bSampled |
                vk::ImageUsageFlagBits::bTransferDst);

        vma::AllocationCreateInfo allocationInfo;
        allocationInfo.setUsage(vma::MemoryUsage::eAutoPreferDevice);

        t.m_Image = m_Gfx->m_Allocator
                        .createImage(imageInfo, allocationInfo, &t.m_Allocation)
                        .get();

        const auto imageViewInfo = vx::imageViewCreateInfo(
            vk::ImageViewType::e2D, t.m_Image, imageInfo.getFormat(),
            vx::subresourceRange(vk::ImageAspectFlagBits::bColor));
        t.m_ImageView = m_Gfx->m_Device.createImageView(imageViewInfo).get();
    }

    EndLayerRendering();

    const auto topLayer = m_SurfaceManager.GetTopLayerHandle();
    const auto postprocessImage = GetPostprocess(PostprocessPrimary()).m_Image;
    ResolveLayer(topLayer, postprocessImage);

    vx::ImageMemoryBarrierState imageBarriers[2];
    auto& [srcImageBarrier, dstImageBarrier] = imageBarriers;

    srcImageBarrier.init(postprocessImage,
                         vx::subresourceRange(vk::ImageAspectFlagBits::bColor));
    srcImageBarrier.setOldLayout(vk::ImageLayout::eTransferDstOptimal);
    srcImageBarrier.setSrcStageAccess(vk::PipelineStageFlagBits2::bTransfer,
                                      vk::AccessFlagBits2::bTransferWrite);
    srcImageBarrier.setNewLayout(vk::ImageLayout::eTransferSrcOptimal);
    srcImageBarrier.setDstStageAccess(vk::PipelineStageFlagBits2::bTransfer,
                                      vk::AccessFlagBits2::bTransferRead);

    dstImageBarrier.init(t.m_Image,
                         vx::subresourceRange(vk::ImageAspectFlagBits::bColor));
    dstImageBarrier.setNewLayout(vk::ImageLayout::eTransferDstOptimal);
    dstImageBarrier.setDstStageAccess(vk::PipelineStageFlagBits2::bTransfer,
                                      vk::AccessFlagBits2::bTransferWrite);
    m_CommandBuffer.cmdPipelineBarriers(rawIns(imageBarriers));

    // Move to origin.
    const vx::Range3D srcRegion(vx::toOffset3D(m_Scissor.getOffset()),
                                vx::toExtent3D(m_Scissor.getExtent()));
    const vx::Range3D dstRegion(vk::Offset3D(),
                                vx::toExtent3D(m_Scissor.getExtent()));

    m_CommandBuffer.cmdCopyImage(
        srcImageBarrier.getImage(), srcRegion, srcImageBarrier.getNewLayout(),
        dstImageBarrier.getImage(), dstRegion, dstImageBarrier.getNewLayout(),
        vx::subresourceLayers(vk::ImageAspectFlagBits::bColor, 1));

    dstImageBarrier.updateLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
    dstImageBarrier.updateStageAccess(
        vk::PipelineStageFlagBits2::bFragmentShader,
        vk::AccessFlagBits2::bShaderRead);
    m_CommandBuffer.cmdPipelineBarriers(dstImageBarrier);

    BeginLayerRendering(topLayer);

    return ~m_TextureResources.Create(t);
}

enum class FilterType { Passthrough, Blur, DropShadow, ColorMatrix, MaskImage };

struct Renderer_VX::FilterBase {
    FilterType type;

    constexpr FilterBase(FilterType type) : type(type) {}
};

struct Renderer_VX::PassthroughFilter : FilterBase {
    constexpr PassthroughFilter() : FilterBase(FilterType::Passthrough) {}
    float blend_factor;
};

struct Renderer_VX::BlurFilter : FilterBase {
    constexpr BlurFilter() : FilterBase(FilterType::Blur) {}
    float sigma;
};

struct Renderer_VX::DropShadowFilter : FilterBase {
    DropShadowFilter() noexcept : FilterBase(FilterType::DropShadow) {}
    float sigma;
    Rml::Vector2f offset;
    Rml::ColourbPremultiplied color;
};

struct Renderer_VX::ColorMatrixFilter : FilterBase {
    ColorMatrixFilter() noexcept : FilterBase(FilterType::ColorMatrix) {}
    Rml::Matrix4f colorMatrix;
};

struct Renderer_VX::MaskImageFilter : FilterBase {
    constexpr MaskImageFilter() : FilterBase(FilterType::MaskImage) {}
};

template<class T>
inline Rml::CompiledFilterHandle CreateFilter(T&& filter) {
    return reinterpret_cast<Rml::CompiledFilterHandle>(
        new T(std::move(filter)));
}

Rml::CompiledFilterHandle Renderer_VX::SaveLayerAsMaskImage() {
    EndLayerRendering();

    const auto topLayer = m_SurfaceManager.GetTopLayerHandle();
    const auto maskImage = GetPostprocess(3).m_Image;
    ResolveLayer(topLayer, maskImage);

    TransitionToSample(maskImage, true);

    BeginLayerRendering(topLayer);

    return CreateFilter(MaskImageFilter());
}

Rml::CompiledFilterHandle
Renderer_VX::CompileFilter(const Rml::String& name,
                           const Rml::Dictionary& parameters) {
    if (name == "opacity") {
        PassthroughFilter filter;
        filter.blend_factor = Rml::Get(parameters, "value", 1.0f);
        return CreateFilter(std::move(filter));
    } else if (name == "blur") {
        BlurFilter filter;
        filter.sigma = Rml::Get(parameters, "sigma", 1.0f);
        return CreateFilter(std::move(filter));
    } else if (name == "drop-shadow") {
        DropShadowFilter filter;
        filter.sigma = Rml::Get(parameters, "sigma", 0.f);
        filter.color =
            Rml::Get(parameters, "color", Rml::Colourb()).ToPremultiplied();
        filter.offset = Rml::Get(parameters, "offset", Rml::Vector2f(0.f));
        return CreateFilter(std::move(filter));
    } else if (name == "brightness") {
        ColorMatrixFilter filter;
        const float value = Rml::Get(parameters, "value", 1.0f);
        filter.colorMatrix = Rml::Matrix4f::Diag(value, value, value, 1.f);
        return CreateFilter(std::move(filter));
    } else if (name == "contrast") {
        ColorMatrixFilter filter;
        const float value = Rml::Get(parameters, "value", 1.0f);
        const float grayness = 0.5f - 0.5f * value;
        filter.colorMatrix = Rml::Matrix4f::Diag(value, value, value, 1.f);
        filter.colorMatrix.SetColumn(
            3, Rml::Vector4f(grayness, grayness, grayness, 1.f));
        return CreateFilter(std::move(filter));
    } else if (name == "invert") {
        ColorMatrixFilter filter;
        const float value =
            Rml::Math::Clamp(Rml::Get(parameters, "value", 1.0f), 0.f, 1.f);
        const float inverted = 1.f - 2.f * value;
        filter.colorMatrix =
            Rml::Matrix4f::Diag(inverted, inverted, inverted, 1.f);
        filter.colorMatrix.SetColumn(3,
                                     Rml::Vector4f(value, value, value, 1.f));
        return CreateFilter(std::move(filter));
    } else if (name == "grayscale") {
        ColorMatrixFilter filter;
        const float value = Rml::Get(parameters, "value", 1.0f);
        const float rev_value = 1.f - value;
        const Rml::Vector3f gray =
            value * Rml::Vector3f(0.2126f, 0.7152f, 0.0722f);
        // clang-format off
        filter.colorMatrix = Rml::Matrix4f::FromRows(
            {gray.x + rev_value, gray.y,             gray.z,             0.f},
            {gray.x,             gray.y + rev_value, gray.z,             0.f},
            {gray.x,             gray.y,             gray.z + rev_value, 0.f},
            {0.f,                0.f,                0.f,                1.f}
        );
        // clang-format on
        return CreateFilter(std::move(filter));
    } else if (name == "sepia") {
        ColorMatrixFilter filter;
        const float value = Rml::Get(parameters, "value", 1.0f);
        const float rev_value = 1.f - value;
        const Rml::Vector3f r_mix =
            value * Rml::Vector3f(0.393f, 0.769f, 0.189f);
        const Rml::Vector3f g_mix =
            value * Rml::Vector3f(0.349f, 0.686f, 0.168f);
        const Rml::Vector3f b_mix =
            value * Rml::Vector3f(0.272f, 0.534f, 0.131f);
        // clang-format off
        filter.colorMatrix = Rml::Matrix4f::FromRows(
            {r_mix.x + rev_value, r_mix.y,             r_mix.z,             0.f},
            {g_mix.x,             g_mix.y + rev_value, g_mix.z,             0.f},
            {b_mix.x,             b_mix.y,             b_mix.z + rev_value, 0.f},
            {0.f,                 0.f,                 0.f,                 1.f}
        );
        // clang-format on
        return CreateFilter(std::move(filter));
    } else if (name == "hue-rotate") {
        // Hue-rotation and saturation values based on:
        // https://www.w3.org/TR/filter-effects-1/#attr-valuedef-type-huerotate
        ColorMatrixFilter filter;
        const float value = Rml::Get(parameters, "value", 1.0f);
        const float s = Rml::Math::Sin(value);
        const float c = Rml::Math::Cos(value);
        // clang-format off
        filter.colorMatrix = Rml::Matrix4f::FromRows(
            {0.213f + 0.787f * c - 0.213f * s,  0.715f - 0.715f * c - 0.715f * s,  0.072f - 0.072f * c + 0.928f * s,  0.f},
            {0.213f - 0.213f * c + 0.143f * s,  0.715f + 0.285f * c + 0.140f * s,  0.072f - 0.072f * c - 0.283f * s,  0.f},
            {0.213f - 0.213f * c - 0.787f * s,  0.715f - 0.715f * c + 0.715f * s,  0.072f + 0.928f * c + 0.072f * s,  0.f},
            {0.f,                               0.f,                               0.f,                               1.f}
        );
        // clang-format on
        return CreateFilter(std::move(filter));
    } else if (name == "saturate") {
        ColorMatrixFilter filter;
        const float value = Rml::Get(parameters, "value", 1.0f);
        // clang-format off
        filter.colorMatrix = Rml::Matrix4f::FromRows(
            {0.213f + 0.787f * value,  0.715f - 0.715f * value,  0.072f - 0.072f * value,  0.f},
            {0.213f - 0.213f * value,  0.715f + 0.285f * value,  0.072f - 0.072f * value,  0.f},
            {0.213f - 0.213f * value,  0.715f - 0.715f * value,  0.072f + 0.928f * value,  0.f},
            {0.f,                      0.f,                      0.f,                      1.f}
        );
        // clang-format on
        return CreateFilter(std::move(filter));
    }

    Rml::Log::Message(Rml::Log::LT_WARNING, "Unsupported filter type '%s'.",
                      name.c_str());
    return {};
}

template<class F>
void Renderer_VX::VisitFilter(FilterBase* p, F f) {
    switch (p->type) {
    case FilterType::Passthrough: return f(static_cast<PassthroughFilter*>(p));
    case FilterType::Blur: return f(static_cast<BlurFilter*>(p));
    case FilterType::DropShadow: return f(static_cast<DropShadowFilter*>(p));
    case FilterType::ColorMatrix: return f(static_cast<ColorMatrixFilter*>(p));
    case FilterType::MaskImage: return f(static_cast<MaskImageFilter*>(p));
    }
}

void Renderer_VX::ReleaseFilter(Rml::CompiledFilterHandle filter) {
    VisitFilter(reinterpret_cast<FilterBase*>(filter),
                [](auto* p) { delete p; });
}

void Renderer_VX::InitPipelineLayouts() {
    const auto device = m_Gfx->m_Device;

    struct Context {
        vk::Sampler immutableSamplers[1];
    };
    Context context{m_Sampler};

    check(device.createTypedDescriptorSetLayoutWithContext(
        &m_TextureDescriptorSetLayout, context,
        vk::DescriptorSetLayoutCreateFlagBits::bPushDescriptorKHR));

    check(device.createTypedDescriptorSetLayout(
        &m_UniformDescriptorSetLayout,
        vk::DescriptorSetLayoutCreateFlagBits::bPushDescriptorKHR));

    check(device.createTypedDescriptorSetLayoutWithContext(
        &m_BlurDescriptorSetLayout, context,
        vk::DescriptorSetLayoutCreateFlagBits::bPushDescriptorKHR));

    check(device.createTypedDescriptorSetLayoutWithContext(
        &m_FilterDescriptorSetLayout, context,
        vk::DescriptorSetLayoutCreateFlagBits::bUpdateAfterBindPool));

    vk::PushConstantRange pushConstantRanges[2];
    auto& [vertPushConstantRange, fragPushConstantRange] = pushConstantRanges;
    vertPushConstantRange.setStageFlags(vk::ShaderStageFlagBits::bVertex);
    vertPushConstantRange.setOffset(0);
    vertPushConstantRange.setSize(sizeof(VsInput));

    fragPushConstantRange.setStageFlags(vk::ShaderStageFlagBits::bFragment);
    fragPushConstantRange.setOffset(64);
    fragPushConstantRange.setSize(4);

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    pipelineLayoutInfo.setPushConstantRangeCount(1);
    pipelineLayoutInfo.setPushConstantRanges(&vertPushConstantRange);

    m_BasicPipelineLayout =
        device.createPipelineLayout(pipelineLayoutInfo).get();

    pipelineLayoutInfo.setSetLayoutCount(1);

    pipelineLayoutInfo.setSetLayouts(&m_UniformDescriptorSetLayout);

    m_GradientPipelineLayout =
        device.createPipelineLayout(pipelineLayoutInfo).get();

    pipelineLayoutInfo.setSetLayouts(&m_TextureDescriptorSetLayout);

    m_TexturePipelineLayout =
        device.createPipelineLayout(pipelineLayoutInfo).get();

    pipelineLayoutInfo.setSetLayouts(&m_FilterDescriptorSetLayout);

    pipelineLayoutInfo.setPushConstantRanges(&fragPushConstantRange);

    m_FilterPipelineLayout =
        device.createPipelineLayout(pipelineLayoutInfo).get();

    vk::DescriptorSetLayout descriptorSetLayouts[] = {
        m_FilterDescriptorSetLayout, m_UniformDescriptorSetLayout};
    pipelineLayoutInfo.setSetLayoutCount(2);
    pipelineLayoutInfo.setSetLayouts(descriptorSetLayouts);

    m_ColorMatrixPipelineLayout =
        device.createPipelineLayout(pipelineLayoutInfo).get();

    pipelineLayoutInfo.setPushConstantRanges(&vertPushConstantRange);

    pipelineLayoutInfo.setSetLayoutCount(1);
    pipelineLayoutInfo.setSetLayouts(&m_BlurDescriptorSetLayout);

    m_BlurPipelineLayout =
        device.createPipelineLayout(pipelineLayoutInfo).get();
}

void Renderer_VX::InitPipelines(
    vk::PipelineRenderingCreateInfo& renderingInfo) {
    const auto device = m_Gfx->m_Device;

    const auto clipVertShader =
        device.createShaderModule(shader_vert_clip).get();
    const auto mainVertShader =
        device.createShaderModule(shader_vert_main).get();
    const auto passthroughVertShader =
        device.createShaderModule(shader_vert_passthrough).get();
    const auto blurVertShader =
        device.createShaderModule(shader_vert_blur).get();
    const auto colorFragShader =
        device.createShaderModule(shader_frag_color).get();
    const auto textureFragShader =
        device.createShaderModule(shader_frag_texture).get();
    const auto gradientFragShader =
        device.createShaderModule(shader_frag_gradient).get();
    const auto passthroughFragShader =
        device.createShaderModule(shader_frag_passthrough).get();
    const auto blurFragShader =
        device.createShaderModule(shader_frag_blur).get();
    const auto colorMatrixFragShader =
        device.createShaderModule(shader_frag_color_matrix).get();
    const auto blendMaskFragShader =
        device.createShaderModule(shader_frag_blend_mask).get();

    vx::GraphicsPipelineBuilder pipelineBuilder;
    pipelineBuilder.attach(renderingInfo);
    pipelineBuilder.setTopology(vk::PrimitiveTopology::eTriangleList);
    pipelineBuilder.setPolygonMode(vk::PolygonMode::eFill);
    pipelineBuilder.setFrontFace(vk::FrontFace::eClockwise);
    pipelineBuilder.setCullMode(vk::CullModeFlagBits::eNone);
    pipelineBuilder.setRasterizationSamples(m_Gfx->m_SampleCount);

    vk::PipelineShaderStageCreateInfo shaderStageInfos[2];
    pipelineBuilder.setStages(shaderStageInfos);

    vk::DynamicState dynamicStates[7];
    pipelineBuilder.setDynamicStates(dynamicStates);

    vk::VertexInputAttributeDescription vertexAttributeDescriptions[3];
    pipelineBuilder.setVertexAttributeDescriptions(vertexAttributeDescriptions);

    shaderStageInfos[0] = vx::makePipelineShaderStageCreateInfo(
        vk::ShaderStageFlagBits::bVertex, {});
    pipelineBuilder.setStageCount(1);

    dynamicStates[0] = vk::DynamicState::eViewport;
    dynamicStates[1] = vk::DynamicState::eScissor;
    dynamicStates[2] = vk::DynamicState::eStencilReference;
    dynamicStates[3] = vk::DynamicState::eStencilOp;
    pipelineBuilder.setDynamicStateCount(4);

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

    dynamicStates[3] = vk::DynamicState::eStencilTestEnable;

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

    pipelineBuilder.setLayout(m_GradientPipelineLayout);
    shaderStageInfos[1].setModule(gradientFragShader);
    m_GradientPipeline = pipelineBuilder.build(device).get();

    pipelineBuilder.setLayout(m_TexturePipelineLayout);
    shaderStageInfos[1].setModule(textureFragShader);

    m_TexturePipeline = pipelineBuilder.build(device).get();

    pipelineBuilder.setLayout(m_FilterPipelineLayout);

    dynamicStates[4] = vk::DynamicState::eBlendConstants;
    dynamicStates[5] = vk::DynamicState::eColorBlendEnableEXT;
    dynamicStates[6] = vk::DynamicState::eColorBlendEquationEXT;
    pipelineBuilder.setDynamicStateCount(7);

    vertexAttributeDescriptions[1].setLocation(1);
    vertexAttributeDescriptions[1].setBinding(0);
    vertexAttributeDescriptions[1].setFormat(vk::Format::eR32G32Sfloat);
    vertexAttributeDescriptions[1].setOffset(offsetof(Rml::Vertex, tex_coord));
    pipelineBuilder.setVertexAttributeDescriptionCount(2);

    shaderStageInfos[0].setModule(passthroughVertShader);
    shaderStageInfos[1].setModule(passthroughFragShader);

    if (m_Gfx->m_SampleCount != vk::SampleCountFlagBits::b1) {
        m_MsPassthroughPipeline = pipelineBuilder.build(device).get();
    }

    pipelineBuilder.setRasterizationSamples(vk::SampleCountFlagBits::b1);
    m_PassthroughPipeline = pipelineBuilder.build(device).get();

    pipelineBuilder.setDynamicStateCount(4);
    pipelineBuilder.setBlendEnable(false);

    shaderStageInfos[1].setModule(colorMatrixFragShader);

    m_ColorMatrixPipeline = pipelineBuilder.build(device).get();

    shaderStageInfos[1].setModule(blendMaskFragShader);

    m_BlendMaskPipeline = pipelineBuilder.build(device).get();

    pipelineBuilder.setLayout(m_BlurPipelineLayout);
    shaderStageInfos[0].setModule(blurVertShader);
    shaderStageInfos[1].setModule(blurFragShader);

    m_BlurPipeline = pipelineBuilder.build(device).get();

    device.destroyShaderModule(clipVertShader);
    device.destroyShaderModule(mainVertShader);
    device.destroyShaderModule(passthroughVertShader);
    device.destroyShaderModule(blurVertShader);
    device.destroyShaderModule(colorFragShader);
    device.destroyShaderModule(textureFragShader);
    device.destroyShaderModule(gradientFragShader);
    device.destroyShaderModule(passthroughFragShader);
    device.destroyShaderModule(blurFragShader);
    device.destroyShaderModule(colorMatrixFragShader);
    device.destroyShaderModule(blendMaskFragShader);
}

Rml::TextureHandle Renderer_VX::CreateTexture(vk::Buffer buffer,
                                              Rml::Vector2i dimensions) {
    const auto device = m_Gfx->m_Device;
    const auto allocator = m_Gfx->m_Allocator;

    TextureResource t;

    const vk::Extent2D extent(dimensions.x, dimensions.y);
    const auto imageInfo =
        vx::image2DCreateInfo(vk::Format::eR8G8B8A8Unorm, extent,
                              vk::ImageUsageFlagBits::bSampled |
                                  vk::ImageUsageFlagBits::bTransferDst);

    vma::AllocationCreateInfo allocationInfo;
    allocationInfo.setUsage(vma::MemoryUsage::eAutoPreferDevice);

    t.m_Image =
        allocator.createImage(imageInfo, allocationInfo, &t.m_Allocation).get();

    const auto commandBuffer = m_Gfx->BeginTemp();

    vx::ImageMemoryBarrierState imageMemoryBarrier;
    imageMemoryBarrier.init(
        t.m_Image, vx::subresourceRange(vk::ImageAspectFlagBits::bColor));

    imageMemoryBarrier.setNewLayout(vk::ImageLayout::eTransferDstOptimal);
    imageMemoryBarrier.setDstStageAccess(vk::PipelineStageFlagBits2::bCopy,
                                         vk::AccessFlagBits2::bTransferWrite);
    commandBuffer.cmdPipelineBarriers(imageMemoryBarrier);

    vk::BufferImageCopy bufferImageCopy;
    bufferImageCopy.setImageSubresource(
        vx::subresourceLayers(vk::ImageAspectFlagBits::bColor, 1));
    bufferImageCopy.setImageExtent(vx::toExtent3D(extent));
    commandBuffer.cmdCopyBufferToImage(buffer, t.m_Image,
                                       vk::ImageLayout::eTransferDstOptimal, 1,
                                       &bufferImageCopy);

    imageMemoryBarrier.updateLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
    imageMemoryBarrier.updateStageAccess(
        vk::PipelineStageFlagBits2::bFragmentShader,
        vk::AccessFlagBits2::bShaderRead);
    commandBuffer.cmdPipelineBarriers(imageMemoryBarrier);

    m_Gfx->EndTemp(commandBuffer);

    const auto imageViewInfo = vx::imageViewCreateInfo(
        vk::ImageViewType::e2D, t.m_Image, imageInfo.getFormat(),
        vx::subresourceRange(vk::ImageAspectFlagBits::bColor));
    t.m_ImageView = device.createImageView(imageViewInfo).get();

    return ~m_TextureResources.Create(t);
}

Renderer_VX::GeometryResource
Renderer_VX::CreateGeometry(Rml::Span<const Rml::Vertex> vertices,
                            Rml::Span<const int> indices) {
    GeometryResource g;
    g.m_VertexCount = uint32_t(vertices.size());
    g.m_IndexCount = uint32_t(indices.size());

    const auto vertexBytes = g.m_VertexCount * sizeof(Rml::Vertex);
    const auto indexBytes = g.m_IndexCount * sizeof(uint32_t);
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
    g.m_Buffer = m_Gfx->m_Allocator
                     .createBuffer(bufferInfo, allocationInfo, &g.m_Allocation,
                                   &allocInfo)
                     .get();

    auto p = static_cast<uint8_t*>(allocInfo.getMappedData());
    std::memcpy(p, vertices.data(), vertexBytes);
    p += vertexBytes;
    std::memcpy(p, indices.data(), indexBytes);

    return g;
}

Renderer_VX::ShaderResource Renderer_VX::CreateShaderResource(const void* data,
                                                              size_t size) {
    ShaderResource s;

    vk::BufferCreateInfo bufferInfo;
    bufferInfo.setSize(size);
    bufferInfo.setUsage(vk::BufferUsageFlagBits::bUniformBuffer);
    bufferInfo.setSharingMode(vk::SharingMode::eExclusive);

    vma::AllocationCreateInfo allocationInfo;
    allocationInfo.setFlags(
        vma::AllocationCreateFlagBits::bMapped |
        vma::AllocationCreateFlagBits::bHostAccessSequentialWrite);
    allocationInfo.setUsage(vma::MemoryUsage::eAutoPreferDevice);

    vma::AllocationInfo allocInfo;
    s.m_Buffer = m_Gfx->m_Allocator
                     .createBuffer(bufferInfo, allocationInfo, &s.m_Allocation,
                                   &allocInfo)
                     .get();

    std::memcpy(allocInfo.getMappedData(), data, size);
    return s;
}

void Renderer_VX::RenderFilters(
    Rml::Span<const Rml::CompiledFilterHandle> filterHandles) {
    for (const auto filterHandle : filterHandles) {
        VisitFilter(reinterpret_cast<FilterBase*>(filterHandle),
                    [this](auto* p) { RenderFilter(*p); });
    }
}

void Renderer_VX::RenderFilter(const PassthroughFilter& filter) {
    const auto postprocesImage = BeginPostprocess(PostprocessSecondary());
    SetSample(PostprocessPrimary());

    m_CommandBuffer.cmdBindPipeline(vk::PipelineBindPoint::eGraphics,
                                    m_PassthroughPipeline);

    const vk::Bool32 colorBlendEnable = true;
    m_CommandBuffer.cmdSetColorBlendEnableEXT(0, 1, &colorBlendEnable);
    const auto colorBlendEquation = MakeColorBlendEquation(
        vk::BlendOp::eAdd, vk::BlendFactor::eConstantColor,
        vk::BlendFactor::eZero);
    m_CommandBuffer.cmdSetColorBlendEquationEXT(0, 1, &colorBlendEquation);
    const float blendConstants[4] = {filter.blend_factor, filter.blend_factor,
                                     filter.blend_factor, filter.blend_factor};
    m_CommandBuffer.cmdSetBlendConstants(blendConstants);

    m_GeometryResources.Get(~m_FullscreenQuadGeometry).Draw(m_CommandBuffer);
    m_CommandBuffer.cmdEndRendering();
    TransitionToSample(postprocesImage, false);
    SwapPostprocessPrimarySecondary();
}

void Renderer_VX::RenderFilter(const BlurFilter& filter) {
    // RenderBlur(filter.sigma, {0, 1});
}

void Renderer_VX::RenderFilter(const ColorMatrixFilter& filter) {
    const auto postprocesImage = BeginPostprocess(PostprocessSecondary());
    SetSample(PostprocessPrimary());

    m_CommandBuffer.cmdBindPipeline(vk::PipelineBindPoint::eGraphics,
                                    m_ColorMatrixPipeline);

    m_CommandBuffer.cmdPushConstant(
        m_FilterPipelineLayout, vk::ShaderStageFlagBits::bFragment,
        VX_FIELD(ColorMatrixFsInput, colorMatrix) = filter.colorMatrix);

    m_GeometryResources.Get(~m_FullscreenQuadGeometry).Draw(m_CommandBuffer);
    m_CommandBuffer.cmdEndRendering();
    TransitionToSample(postprocesImage, false);
    SwapPostprocessPrimarySecondary();
}

void Renderer_VX::RenderFilter(const MaskImageFilter&) {
    const auto postprocesImage = BeginPostprocess(PostprocessSecondary());
    SetSample(PostprocessPrimary());

    m_CommandBuffer.cmdBindPipeline(vk::PipelineBindPoint::eGraphics,
                                    m_BlendMaskPipeline);

    m_GeometryResources.Get(~m_FullscreenQuadGeometry).Draw(m_CommandBuffer);
    m_CommandBuffer.cmdEndRendering();
    TransitionToSample(postprocesImage, false);
    SwapPostprocessPrimarySecondary();
}

#if 0
void Renderer_VX::RenderBlur(float sigma, const unsigned(&postprocess)[2]) {
    constexpr int max_num_passes = 10;
    constexpr float max_single_pass_sigma = 3.0f;
    const int pass_level = Rml::Math::Clamp(
        Rml::Math::Log2(int(sigma * (2.f / max_single_pass_sigma))), 0,
        max_num_passes);
    sigma = Rml::Math::Clamp(sigma / float(1 << pass_level), 0.0f,
        max_single_pass_sigma);

    const auto extent = m_Gfx->m_FrameExtent;

    // Begin by downscaling so that the blur pass can be done at a reduced
    // resolution for large sigma.
    auto scissor = m_Scissor;
    m_CommandBuffer.cmdBindPipeline(vk::PipelineBindPoint::eGraphics,
        m_PassthroughPipeline);
    // m_CommandBuffer.cmdSetScissor(0, 1, &scissor);
    const vk::Bool32 colorBlendEnable = false;
    m_CommandBuffer.cmdSetColorBlendEnableEXT(0, 1, &colorBlendEnable);

    // Downscale by iterative half-scaling with bilinear filtering, to reduce
    // aliasing.
    SetViewport(m_CommandBuffer, extent.width / 2, extent.height / 2);

    // Scale UVs if we have even dimensions, such that texture fetches align
    // perfectly between texels, thereby producing a 50% blend of neighboring
    // texels.
    const Rml::Vector2f uv_scaling = {
        (extent.width & 1) ? (1.f - 1.f / extent.width) : 1.f,
        (extent.height & 1) ? (1.f - 1.f / extent.height) : 1.f};

    const auto quadGeometry = UseFullscreenQuad({}, uv_scaling);

    for (int i = 0; i < pass_level; ++i) {
        scissor.offset.x = (scissor.offset.x + 1) / 2;
        scissor.offset.y = (scissor.offset.y + 1) / 2;
        scissor.extent.width /= 2;
        scissor.extent.height /= 2;
        const auto j = i & 1;
        const auto postprocesImage = BeginPostprocess(postprocess[j ^ 1]);
        SetSample(postprocess[j]);
        m_CommandBuffer.cmdSetScissor(0, 1, &scissor);
        m_GeometryResources.Get(~quadGeometry).Draw(m_CommandBuffer);
        m_CommandBuffer.cmdEndRendering();
        TransitionToSample(postprocesImage, false);
    }

    SetViewport(m_CommandBuffer, extent.width, extent.height);

    // Ensure texture data end up in the temp buffer. Depending on the last
    // downscaling, we might need to move it from the source_destination buffer.
    if ((pass_level & 1) == 0) {
        const auto postprocesImage = BeginPostprocess(postprocess[1]);
        SetSample(postprocess[0]);
        m_GeometryResources.Get(~m_FullscreenQuadGeometry)
            .Draw(m_CommandBuffer);
        m_CommandBuffer.cmdEndRendering();
        TransitionToSample(postprocesImage, false);
    }

    m_CommandBuffer.cmdBindPipeline(vk::PipelineBindPoint::eGraphics,
        m_BlurPipeline);
    // Set up uniforms.
    {
        BlurFsInput input;
        input.SetBlurWeights(sigma);
        input.SetTexCoordLimits(scissor, extent);

        const uint8_t useFlag = 2u << m_Gfx->m_FrameNumber;
        const auto resource = m_ShaderResources.Create(
            CreateShaderResource(&input, sizeof(input)), useFlag);

        const vx::DescriptorSet<BlurDescriptorSet> descriptorSet;
        m_CommandBuffer.cmdPushDescriptorSetKHR(
            vk::PipelineBindPoint::eGraphics, m_BlurPipelineLayout, 0,
            {descriptorSet->input = vx::UniformBufferDescriptor(
                 m_ShaderResources.Get(resource).m_Buffer)});
    }

    // Blur render pass - vertical.
    const auto postprocesImage0 = BeginPostprocess(postprocess[0]);
    SetSample(postprocess[1], m_BlurPipelineLayout);

    m_CommandBuffer.cmdPushConstant(
        m_BlurPipelineLayout, vk::ShaderStageFlagBits::bVertex,
        VX_FIELD(BlurVsInput, texelOffset) = {0.f, 1.f / extent.height});
    m_GeometryResources.Get(~m_FullscreenQuadGeometry).Draw(m_CommandBuffer);
    m_CommandBuffer.cmdEndRendering();
    TransitionToSample(postprocesImage0, false);

    // Blur render pass - horizontal.
    const auto postprocesImage1 = BeginPostprocess(postprocess[1]);
    SetSample(postprocess[0], m_BlurPipelineLayout);

    m_CommandBuffer.cmdPushConstant(
        m_BlurPipelineLayout, vk::ShaderStageFlagBits::bVertex,
        VX_FIELD(BlurVsInput, texelOffset) = {1.f / extent.width, 0.f});
    m_GeometryResources.Get(~m_FullscreenQuadGeometry).Draw(m_CommandBuffer);
    m_CommandBuffer.cmdEndRendering();

    // Blit the blurred image to the scissor region with upscaling.
    m_CommandBuffer.cmdSetScissor(0, 1, &m_Scissor);

    vx::ImageMemoryBarrierState imageBarriers[2];
    auto& [srcImageBarrier, dstImageBarrier] = imageBarriers;

    srcImageBarrier.init(postprocesImage1,
        vx::subresourceRange(vk::ImageAspectFlagBits::bColor));
    srcImageBarrier.setOldLayout(vk::ImageLayout::eAttachmentOptimal);
    srcImageBarrier.setSrcStageAccess(
        vk::PipelineStageFlagBits2::bColorAttachmentOutput,
        vk::AccessFlagBits2::bColorAttachmentWrite);
    srcImageBarrier.setNewLayout(vk::ImageLayout::eTransferSrcOptimal);
    srcImageBarrier.setDstStageAccess(vk::PipelineStageFlagBits2::bTransfer,
        vk::AccessFlagBits2::bTransferRead);

    dstImageBarrier.init(postprocesImage0,
        vx::subresourceRange(vk::ImageAspectFlagBits::bColor));
    dstImageBarrier.setNewLayout(vk::ImageLayout::eTransferDstOptimal);
    dstImageBarrier.setDstStageAccess(vk::PipelineStageFlagBits2::bTransfer,
        vk::AccessFlagBits2::bTransferWrite);
    m_CommandBuffer.cmdPipelineBarriers(rawIns(imageBarriers));

    const vx::Range3D srcRegion(vx::toOffset3D(scissor.getOffset()),
        vx::toExtent3D(scissor.getExtent()));
    const vx::Range3D dstRegion(vx::toOffset3D(m_Scissor.getOffset()),
        vx::toExtent3D(m_Scissor.getExtent()));

    m_CommandBuffer.cmdBlitImage(
        srcImageBarrier.getImage(), srcRegion, srcImageBarrier.getNewLayout(),
        dstImageBarrier.getImage(), dstRegion, dstImageBarrier.getNewLayout(),
        vx::subresourceLayers(vk::ImageAspectFlagBits::bColor, 1),
        vk::Filter::eLinear);

    // The above upscale blit might be jittery at low resolutions (large pass
    // levels). This is especially noticeable when moving an element with
    // backdrop blur around or when trying to click/hover an element within a
    // blurred region since it may be rendered at an offset. For more stable and
    // accurate rendering we next upscale the blur image by an exact
    // power-of-two. However, this may not fill the edges completely so we need
    // to do the above first. Note that this strategy may sometimes result in
    // visible seams. Alternatively, we could try to enlarge the window to the
    // next power-of-two size and then downsample and blur that.
    const auto scale = 1 << pass_level;
    const vx::Range3D targetRegion(
        vk::Offset3D(srcRegion.min.x * scale, srcRegion.min.y * scale,
            srcRegion.min.z),
        vk::Offset3D(srcRegion.max.x * scale, srcRegion.max.y * scale,
            srcRegion.max.z));
    if (dstRegion != targetRegion) {
        m_CommandBuffer.cmdBlitImage(
            srcImageBarrier.getImage(), srcRegion,
            srcImageBarrier.getNewLayout(), dstImageBarrier.getImage(),
            targetRegion, dstImageBarrier.getNewLayout(),
            vx::subresourceLayers(vk::ImageAspectFlagBits::bColor, 1),
            vk::Filter::eLinear);
    }

    TransitionToSample(postprocesImage0, true);
}
#endif // 0
