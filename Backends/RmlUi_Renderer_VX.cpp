#include "RmlUi_Renderer_VX.h"
#include <RmlUi/Core/Core.h>
#include <RmlUi/Core/FileInterface.h>
#include <RmlUi/Core/Log.h>
#include "RmlUi_VX/ShadersCompiledSPV.h"
#include <boost/unordered/unordered_flat_set.hpp>

#define FIELD(s, m) offsetof(s, m), sizeof(s::m)

struct Renderer_VX::MyDescriptorSet {
    static constexpr vk::ShaderStageFlags Stages =
        vk::ShaderStageFlagBits::bFragment;

    VX_BINDING(0, vx::CombinedImageSamplerDescriptor, Stages) tex;
};

struct ResourceBase {
    uint32_t m_RefCount = 1;
};

struct GeometryResource : ResourceBase {
    uint32_t m_VertexCount;
    uint32_t m_IndexCount;
    vk::Buffer m_Buffer;
    vma::Allocation m_Allocation;
};

struct TextureResource : ResourceBase {
    vk::Image m_Image;
    vk::ImageView m_ImageView;
    vma::Allocation m_Allocation;
};

struct Renderer_VX::FrameResources {
    boost::unordered_flat_set<Rml::CompiledGeometryHandle> m_Geometries;
    boost::unordered_flat_set<Rml::TextureHandle> m_Textures;
};

struct StagingBuffer {
    vma::Allocator m_Allocator;
    // Rresult
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

struct VsInput {
    Rml::Matrix4f transform;
    Rml::Vector2f translate;
};

Renderer_VX::Renderer_VX() = default;
Renderer_VX::~Renderer_VX() = default;

bool Renderer_VX::Init(const Backend& backend, uint32_t frameCount) {
    m_Backend = &backend;
    m_FrameResources.reset(new FrameResources[frameCount]);

    const auto device = m_Backend->GetDevice(this);
    check(device.createTypedDescriptorSetLayout(
        &m_DescriptorSetLayout,
        vk::DescriptorSetLayoutCreateFlagBits::bPushDescriptorKHR));

    InitPipelines();

    vk::SamplerCreateInfo samplerInfo;
    samplerInfo.setMagFilter(vk::Filter::eLinear);
    samplerInfo.setMinFilter(vk::Filter::eLinear);
    samplerInfo.setAddressModeU(vk::SamplerAddressMode::eRepeat);
    samplerInfo.setAddressModeV(vk::SamplerAddressMode::eRepeat);
    samplerInfo.setAddressModeW(vk::SamplerAddressMode::eRepeat);
    m_Sampler = device.createSampler(samplerInfo).get();

    return true;
}

void Renderer_VX::Shutdown() {
    const auto device = m_Backend->GetDevice(this);

    if (m_Sampler) {
        device.destroySampler(m_Sampler);
    }
    for (const auto pipeline : m_Pipelines) {
        if (pipeline) {
            device.destroyPipeline(pipeline);
        }
    }
    for (const auto pipelineLayout : m_PipelineLayouts) {
        if (pipelineLayout) {
            device.destroyPipelineLayout(pipelineLayout);
        }
    }
    if (m_DescriptorSetLayout) {
        device.destroyDescriptorSetLayout(m_DescriptorSetLayout);
    }
}

Rml::Matrix4f GetMvp(vk::Extent2D extent, const Rml::Matrix4f& transform) {
    auto projection = Rml::Matrix4f::ProjectOrtho(
        0.0f, float(extent.getWidth()), float(extent.getHeight()), 0.0f, -10000,
        10000);

    // https://matthewwellings.com/blog/the-new-vulkan-coordinate-system/
    Rml::Matrix4f correction_matrix;
    correction_matrix.SetColumns(Rml::Vector4f(1.0f, 0.0f, 0.0f, 0.0f),
                                 Rml::Vector4f(0.0f, -1.0f, 0.0f, 0.0f),
                                 Rml::Vector4f(0.0f, 0.0f, 0.5f, 0.0f),
                                 Rml::Vector4f(0.0f, 0.0f, 0.5f, 1.0f));

    projection = correction_matrix * projection;

    return projection * transform;
}

void Renderer_VX::BeginFrame(vx::CommandBuffer commandBuffer, uint32_t frame) {
    m_CommandBuffer = commandBuffer;
    m_FrameNumber = frame;

    const auto extent = m_Backend->GetFrameExtent(this);
    const vk::Rect2D renderArea{{}, extent};
    m_CommandBuffer.cmdSetScissor(0, 1, &renderArea);

    const VsInput vsInput{.transform =
                              GetMvp(extent, Rml::Matrix4f::Identity())};
    m_CommandBuffer.cmdPushConstants(m_PipelineLayouts[ColorPipeline],
                                     vk::ShaderStageFlagBits::bVertex, 0,
                                     sizeof(vsInput), &vsInput);
}

void Renderer_VX::EndFrame() { m_CommandBuffer = {}; }

void Renderer_VX::ResetFrame(uint32_t frame) {
    auto& frameResources = m_FrameResources[frame];
    for (const auto g : frameResources.m_Geometries) {
        ReleaseGeometry(g);
    }
    frameResources.m_Geometries.clear();
    for (const auto t : frameResources.m_Textures) {
        ReleaseTexture(t);
    }
    frameResources.m_Textures.clear();
}

Rml::CompiledGeometryHandle
Renderer_VX::CompileGeometry(Rml::Span<const Rml::Vertex> vertices,
                             Rml::Span<const int> indices) {
    const auto allocator = m_Backend->GetAllocator(this);

    auto g = std::make_unique<GeometryResource>();
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

    return reinterpret_cast<Rml::CompiledGeometryHandle>(g.release());
}

void Renderer_VX::RenderGeometry(Rml::CompiledGeometryHandle handle,
                                 Rml::Vector2f translation,
                                 Rml::TextureHandle texture) {
    auto& frameResources = m_FrameResources[m_FrameNumber];
    const auto g = reinterpret_cast<GeometryResource*>(handle);
    if (frameResources.m_Geometries.insert(handle).second) {
        ++g->m_RefCount;
    }
    auto pipeline = ColorPipeline;
    if (texture) {
        pipeline = TexturePipeline;
        const auto t = reinterpret_cast<TextureResource*>(texture);
        if (frameResources.m_Textures.insert(texture).second) {
            ++t->m_RefCount;
        }
        const vx::DescriptorSet<MyDescriptorSet> descriptorSet;
        m_CommandBuffer.cmdPushTypedDescriptorSetKHR(
            vk::PipelineBindPoint::eGraphics, m_PipelineLayouts[pipeline], 0,
            descriptorSet->tex = vx::CombinedImageSamplerDescriptor(
                m_Sampler, t->m_ImageView,
                vk::ImageLayout::eShaderReadOnlyOptimal));
    }
    m_CommandBuffer.cmdBindPipeline(vk::PipelineBindPoint::eGraphics,
                                    m_Pipelines[pipeline]);
    m_CommandBuffer.cmdPushConstants(m_PipelineLayouts[pipeline],
                                     vk::ShaderStageFlagBits::bVertex,
                                     FIELD(VsInput, translate), &translation);
    const vk::DeviceSize offset = 0;
    m_CommandBuffer.cmdBindVertexBuffers(0, 1, &g->m_Buffer, &offset);
    m_CommandBuffer.cmdBindIndexBuffer(g->m_Buffer,
                                       g->m_VertexCount * sizeof(Rml::Vertex),
                                       vk::IndexType::eUint32);
    m_CommandBuffer.cmdDrawIndexed(g->m_IndexCount, 1, 0, 0, 0);
}

void Renderer_VX::ReleaseGeometry(Rml::CompiledGeometryHandle geometry) {
    const auto g = reinterpret_cast<GeometryResource*>(geometry);
    if (--g->m_RefCount) {
        return;
    }
    const auto allocator = m_Backend->GetAllocator(this);
    allocator.destroyBuffer(g->m_Buffer, g->m_Allocation);
    delete g;
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
    StagingBuffer stagingBuffer{m_Backend->GetAllocator(this)};

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
    StagingBuffer stagingBuffer{m_Backend->GetAllocator(this)};
    std::memcpy(stagingBuffer.Alloc(source_data.size()), source_data.data(),
                source_data.size());
    return CreateTexture(stagingBuffer.m_Buffer, source_dimensions);
}

void Renderer_VX::ReleaseTexture(Rml::TextureHandle texture_handle) {
    const auto t = reinterpret_cast<TextureResource*>(texture_handle);
    if (--t->m_RefCount) {
        return;
    }
    const auto device = m_Backend->GetDevice(this);
    const auto allocator = m_Backend->GetAllocator(this);
    device.destroyImageView(t->m_ImageView);
    allocator.destroyImage(t->m_Image, t->m_Allocation);
    delete t;
}

void Renderer_VX::EnableScissorRegion(bool enable) {
    m_EnableScissor = enable;
    if (!m_CommandBuffer) {
        return;
    }
    if (!m_EnableScissor) {
        const vk::Rect2D scissor{{}, m_Backend->GetFrameExtent(this)};
        m_CommandBuffer.cmdSetScissor(0, 1, &scissor);
    }
}

void Renderer_VX::SetScissorRegion(Rml::Rectanglei region) {
    if (!m_EnableScissor) {
        return;
    }
    if (m_HasTransform) {
        // TODO
    } else {
        const auto extent = m_Backend->GetFrameExtent(this);
        region.Intersect(Rml::Rectanglei::FromSize(
            Rml::Vector2i(extent.getWidth(), extent.getHeight())));
        const vk::Rect2D scissor{vk::Offset2D(region.Left(), region.Top()),
                                 vk::Extent2D(region.Width(), region.Height())};
        m_CommandBuffer.cmdSetScissor(0, 1, &scissor);
    }
}

void Renderer_VX::SetTransform(const Rml::Matrix4f* transform) {
    m_HasTransform = transform != nullptr;
    const auto extent = m_Backend->GetFrameExtent(this);
    const auto mat =
        GetMvp(extent, m_HasTransform ? *transform : Rml::Matrix4f::Identity());
    m_CommandBuffer.cmdPushConstants(m_PipelineLayouts[ColorPipeline],
                                     vk::ShaderStageFlagBits::bVertex,
                                     FIELD(VsInput, transform), &mat);
}

inline vk::ShaderModuleCreateInfo shaderSource(std::span<const uint32_t> data) {
    return {data.size_bytes(), data.data()};
};

void Renderer_VX::InitPipelines() {
    const auto device = m_Backend->GetDevice(this);

    vk::PushConstantRange pushConstantRange;
    pushConstantRange.setStageFlags(vk::ShaderStageFlagBits::bVertex);
    pushConstantRange.setOffset(0);
    pushConstantRange.setSize(sizeof(VsInput));

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    pipelineLayoutInfo.setPushConstantRangeCount(1);
    pipelineLayoutInfo.setPushConstantRanges(&pushConstantRange);

    m_PipelineLayouts[ColorPipeline] =
        device.createPipelineLayout(pipelineLayoutInfo).get();

    pipelineLayoutInfo.setSetLayoutCount(1);
    pipelineLayoutInfo.setSetLayouts(&m_DescriptorSetLayout);

    m_PipelineLayouts[TexturePipeline] =
        device.createPipelineLayout(pipelineLayoutInfo).get();

    vx::GraphicsPipelineBuilder pipelineBuilder;
    pipelineBuilder.setLayout(m_PipelineLayouts[ColorPipeline]);
    pipelineBuilder.setRenderPass(m_Backend->GetRenderPass(this));
    pipelineBuilder.setTopology(vk::PrimitiveTopology::eTriangleList);
    pipelineBuilder.setPolygonMode(vk::PolygonMode::eFill);
    pipelineBuilder.setFrontFace(vk::FrontFace::eClockwise);
    pipelineBuilder.setCullMode(vk::CullModeFlagBits::eNone);
    // pipelineBuilder.enableDepthTest();
    // pipelineBuilder.enableDepthWrite();
    pipelineBuilder.setDepthCompareOp(vk::CompareOp::eAlways);
    pipelineBuilder.setDepthBounds(0.f, 1.f);
    pipelineBuilder.setColorWriteMask(
        vk::ColorComponentFlagBits::bR | vk::ColorComponentFlagBits::bG |
        vk::ColorComponentFlagBits::bB | vk::ColorComponentFlagBits::bA);
    pipelineBuilder.enableBlend();
    pipelineBuilder.setColorBlend(vk::BlendOp::eAdd, vk::BlendFactor::eOne,
                                  vk::BlendFactor::eOneMinusSrcAlpha);
    pipelineBuilder.setAlphaBlend(vk::BlendOp::eSubtract, vk::BlendFactor::eOne,
                                  vk::BlendFactor::eOneMinusSrcAlpha);
    // TODO
    // pipelineBuilder.enableStencilTest();

    vk::VertexInputBindingDescription vertexBindingDescriptions[1];
    vertexBindingDescriptions[0].setBinding(0);
    vertexBindingDescriptions[0].setStride(sizeof(Rml::Vertex));
    vertexBindingDescriptions[0].setInputRate(vk::VertexInputRate::eVertex);
    pipelineBuilder.setVertexBindingDescriptions(vertexBindingDescriptions);

    vk::VertexInputAttributeDescription vertexAttributeDescriptions[3];
    vertexAttributeDescriptions[0].setLocation(0);
    vertexAttributeDescriptions[0].setBinding(0);
    vertexAttributeDescriptions[0].setFormat(vk::Format::eR32G32Sfloat);
    vertexAttributeDescriptions[0].setOffset(offsetof(Rml::Vertex, position));
    vertexAttributeDescriptions[1].setLocation(1);
    vertexAttributeDescriptions[1].setBinding(0);
    vertexAttributeDescriptions[1].setFormat(vk::Format::eR8G8B8A8Unorm);
    vertexAttributeDescriptions[1].setOffset(offsetof(Rml::Vertex, colour));
    vertexAttributeDescriptions[2].setLocation(2);
    vertexAttributeDescriptions[2].setBinding(0);
    vertexAttributeDescriptions[2].setFormat(vk::Format::eR32G32Sfloat);
    vertexAttributeDescriptions[2].setOffset(offsetof(Rml::Vertex, tex_coord));
    pipelineBuilder.setVertexAttributeDescriptions(vertexAttributeDescriptions);

    vk::DynamicState dynamicStates[] = {vk::DynamicState::eViewport,
                                        vk::DynamicState::eScissor};
    pipelineBuilder.setDynamicStates(dynamicStates);

    const auto vertShader =
        device.createShaderModule(shaderSource(shader_vert)).get();
    const auto colorFragShader =
        device.createShaderModule(shaderSource(shader_frag_color)).get();
    const auto textureFragShader =
        device.createShaderModule(shaderSource(shader_frag_texture)).get();

    vk::PipelineShaderStageCreateInfo shaderStageInfos[2] = {
        vx::makePipelineShaderStageCreateInfo(vk::ShaderStageFlagBits::bVertex,
                                              vertShader),
        vx::makePipelineShaderStageCreateInfo(
            vk::ShaderStageFlagBits::bFragment, colorFragShader),
    };
    pipelineBuilder.setStages(shaderStageInfos);

    m_Pipelines[ColorPipeline] = pipelineBuilder.build(device).get();

    pipelineBuilder.setLayout(m_PipelineLayouts[TexturePipeline]);
    shaderStageInfos[1].setModule(textureFragShader);

    m_Pipelines[TexturePipeline] = pipelineBuilder.build(device).get();

    device.destroyShaderModule(vertShader);
    device.destroyShaderModule(colorFragShader);
    device.destroyShaderModule(textureFragShader);
}

Rml::TextureHandle Renderer_VX::CreateTexture(vk::Buffer buffer,
                                              Rml::Vector2i dimensions) {
    const vk::Extent2D extent(dimensions.x, dimensions.y);
    const auto device = m_Backend->GetDevice(this);
    const auto allocator = m_Backend->GetAllocator(this);

    auto t = std::make_unique<TextureResource>();

    const auto imageInfo =
        vx::image2DCreateInfo(vk::Format::eR8G8B8A8Unorm, extent,
                              vk::ImageUsageFlagBits::bSampled |
                                  vk::ImageUsageFlagBits::bTransferDst);

    vma::AllocationCreateInfo allocationInfo;
    allocationInfo.setUsage(vma::MemoryUsage::eAutoPreferDevice);

    t->m_Image =
        allocator.createImage(imageInfo, allocationInfo, &t->m_Allocation)
            .get();

    const auto imageViewInfo =
        vx::imageView2DCreateInfo(t->m_Image, vk::Format::eR8G8B8A8Unorm,
                                  vk::ImageAspectFlagBits::bColor);
    t->m_ImageView = device.createImageView(imageViewInfo).get();

    const auto commandBuffer = m_Backend->BeginCommands(this);

    vx::ImageMemoryBarrierState imageMemoryBarrier(
        t->m_Image, vk::ImageAspectFlagBits::bColor);

    imageMemoryBarrier.update(vk::ImageLayout::eTransferDstOptimal,
                              vk::PipelineStageFlagBits2::bCopy,
                              vk::AccessFlagBits2::bTransferWrite);
    commandBuffer.cmdPipelineBarriers(imageMemoryBarrier);

    vk::BufferImageCopy bufferImageCopy;
    bufferImageCopy.setImageSubresource(
        vx::singleSubresourceLayers(vk::ImageAspectFlagBits::bColor));
    bufferImageCopy.setImageExtent(vx::toExtent3D(extent));
    commandBuffer.cmdCopyBufferToImage(buffer, t->m_Image,
                                       vk::ImageLayout::eTransferDstOptimal, 1,
                                       &bufferImageCopy);

    imageMemoryBarrier.update(vk::ImageLayout::eShaderReadOnlyOptimal,
                              vk::PipelineStageFlagBits2::bFragmentShader,
                              vk::AccessFlagBits2::bShaderRead);
    commandBuffer.cmdPipelineBarriers(imageMemoryBarrier);

    m_Backend->EndCommands(this);

    return reinterpret_cast<Rml::TextureHandle>(t.release());
}