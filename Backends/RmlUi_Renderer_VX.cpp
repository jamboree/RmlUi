#include "RmlUi_Renderer_VX.h"
#include <RmlUi/Core/Core.h>
#include <RmlUi/Core/DecorationTypes.h>
#include <RmlUi/Core/SystemInterface.h>
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

struct TexCoordLimits {
    Rml::Vector2f texCoordMin;
    Rml::Vector2f texCoordMax;

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
};

struct DropShadowParams : TexCoordLimits {
    Rml::Colourf color;
};

struct BlurParams : TexCoordLimits {
    float weights[BLUR_NUM_WEIGHTS];

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

struct CreationParams {
    float value;
    Rml::Vector2f dimensions;
};

struct VsInput {
    Rml::Vector2f translate;
    unsigned transformIdx;

    static void SetRange(vk::PushConstantRange& range) {
        range.setOffset(0);
        range.setSize(sizeof(VsInput));
    }
};

struct BlurVsInput {
    Rml::Vector2f texelOffset;
};

struct FsInput {
    uint8_t _pad[12];
    unsigned texIdx;
    union {
        unsigned colorMatrixIdx;
        BlurParams blurParams;
        DropShadowParams dropShadowParams;
    };

    static void SetRange(vk::PushConstantRange& range) {
        range.setOffset(sizeof(_pad));
        range.setSize(sizeof(FsInput) - sizeof(_pad));
    }
};

struct Renderer_VX::BufferResource {
    vk::Buffer m_Buffer;
    vma::Allocation m_Allocation;
};

struct Renderer_VX::GeometryResource : BufferResource {
    uint32_t m_VertexCount;
    uint32_t m_IndexCount;

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

struct Renderer_VX::FrameResource {
    vx::DescriptorSet<PrimaryDescriptorSet> m_PrimaryDescriptorSet; // not own
    BufferResource m_StorageBuffer;
};

struct Renderer_VX::PrimaryDescriptorSet {
    static constexpr auto Vert = vk::ShaderStageFlagBits::bVertex;
    static constexpr auto Frag = vk::ShaderStageFlagBits::bFragment;
    static constexpr auto UpdateAfterBind =
        vk::DescriptorBindingFlagBits::bUpdateAfterBind;
    static constexpr auto PartiallyBound =
        vk::DescriptorBindingFlagBits::bPartiallyBound;

    VX_BINDING(0, vx::StorageBufferDescriptor, Vert | Frag, UpdateAfterBind)
    matrices;
    VX_BINDING(1, vx::ImmutableSamplerDescriptor<0>, Frag) mySampler;
    VX_BINDING(2, vx::SampledImageDescriptor[4], Frag,
               UpdateAfterBind | PartiallyBound)
    textures;
};

struct Renderer_VX::TextureDescriptorSet {
    static constexpr auto Frag = vk::ShaderStageFlagBits::bFragment;

    VX_BINDING(0, vx::CombinedImageImmutableSamplerDescriptor<0>, Frag) tex;
};

struct Renderer_VX::UniformDescriptorSet {
    static constexpr auto Frag = vk::ShaderStageFlagBits::bFragment;

    VX_BINDING(0, vx::UniformBufferDescriptor, Frag) uniform;
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

    InitPipelineLayouts();

    vk::DescriptorPoolCreateInfo descriptorPoolInfo;
    descriptorPoolInfo.setFlags(
        vk::DescriptorPoolCreateFlagBits::bUpdateAfterBind);
    descriptorPoolInfo.setMaxSets(GfxContext_VX::InFlightCount);
    const vk::DescriptorPoolSize poolSizes[] = {
        {vk::DescriptorType::eSampler, 1 * GfxContext_VX::InFlightCount},
        {vk::DescriptorType::eStorageBuffer, 1 * GfxContext_VX::InFlightCount},
        {vk::DescriptorType::eSampledImage, 4 * GfxContext_VX::InFlightCount}};
    descriptorPoolInfo.setPoolSizeCount(std::size(poolSizes));
    descriptorPoolInfo.setPoolSizes(poolSizes);
    m_DescriptorPool = device.createDescriptorPool(descriptorPoolInfo).get();

    m_FrameResources.reset(new FrameResource[GfxContext_VX::InFlightCount]);
    for (auto& frameResource :
         std::span(m_FrameResources.get(), GfxContext_VX::InFlightCount)) {
        frameResource.m_PrimaryDescriptorSet =
            device
                .allocateTypedDescriptorSet<PrimaryDescriptorSet>(
                    m_DescriptorPool, m_PrimaryDescriptorSetLayout)
                .get();
    }

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

    ReleaseAllResourceUse((2u << GfxContext_VX::InFlightCount) - 2u);
    for (auto& frameResource :
         std::span(m_FrameResources.get(), GfxContext_VX::InFlightCount)) {
        ReleaseFrameResource(frameResource);
    }

    m_SurfaceManager.Destroy(*m_Gfx);
    if (m_FullscreenQuadGeometry) {
        ReleaseGeometry(m_FullscreenQuadGeometry);
    }

    if (m_Sampler) {
        device.destroySampler(m_Sampler);
    }
    if (m_DescriptorPool) {
        device.destroyDescriptorPool(m_DescriptorPool);
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
    if (m_DropShadowPipeline) {
        device.destroyPipeline(m_DropShadowPipeline);
    }
    if (m_CreationPipeline) {
        device.destroyPipeline(m_CreationPipeline);
    }
    if (m_PrimaryPipelineLayout) {
        device.destroyPipelineLayout(m_PrimaryPipelineLayout);
    }
    if (m_GradientPipelineLayout) {
        device.destroyPipelineLayout(m_GradientPipelineLayout);
    }
    if (m_TexturePipelineLayout) {
        device.destroyPipelineLayout(m_TexturePipelineLayout);
    }
    if (m_TextureDescriptorSetLayout) {
        device.destroyDescriptorSetLayout(m_TextureDescriptorSetLayout);
    }
    if (m_UniformDescriptorSetLayout) {
        device.destroyDescriptorSetLayout(m_UniformDescriptorSetLayout);
    }
    if (m_PrimaryDescriptorSetLayout) {
        device.destroyDescriptorSetLayout(m_PrimaryDescriptorSetLayout);
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
    m_Matrices.clear();
    m_StencilRef = 1;
    m_PostprocessIndex = 0;
    m_EnableScissor = false;
    m_EnableClipMask = false;
    m_LayerRendering = false;
    m_DepthStencilInitialized = false;

    const auto extent = m_Gfx->m_FrameExtent;
    const auto& frameResource = m_FrameResources[m_Gfx->m_FrameIndex];

    SetViewport(m_CommandBuffer, extent.width, extent.height);

    m_Scissor.setOffset({});
    m_Scissor.setExtent(extent);
    m_CommandBuffer.cmdSetScissor(0, 1, &m_Scissor);

    const auto transform = Project(extent, Rml::Matrix4f::Identity());
    m_CommandBuffer.cmdPushConstant(
        m_PrimaryPipelineLayout, vk::ShaderStageFlagBits::bVertex,
        VX_FIELD(VsInput, transformIdx) = CreateMatrix(transform));
    m_CommandBuffer.cmdSetStencilTestEnable(m_EnableClipMask);
    m_CommandBuffer.cmdSetStencilReference(
        vk::StencilFaceFlagBits::eFrontAndBack, m_StencilRef);

    m_CommandBuffer.cmdBindDescriptorSets(
        vk::PipelineBindPoint::eGraphics, m_PrimaryPipelineLayout, 0, 1,
        &frameResource.m_PrimaryDescriptorSet);

    const auto topLayer = m_SurfaceManager.PushLayer(*m_Gfx);
    BeginLayerRendering(topLayer);
}

void Renderer_VX::EndFrame() {
    RMLUI_ASSERT(m_SurfaceManager.GetTopLayerHandle() == 0);
    const uint8_t useFlag = 2u << m_Gfx->m_FrameIndex;
    auto& frameResource = m_FrameResources[m_Gfx->m_FrameIndex];

    EndLayerRendering();
    ResolveLayer(0, m_Gfx->CurrentPresentResource().m_Image);
    m_SurfaceManager.PopLayer();

    const auto size = m_Matrices.size() * sizeof(Rml::Matrix4f);
    vma::AllocationInfo allocInfo;
    frameResource.m_StorageBuffer = CreateBufferResource(
        size, vk::BufferUsageFlagBits::bStorageBuffer, &allocInfo);
    std::memcpy(allocInfo.getMappedData(), m_Matrices.data(), size);

    m_Gfx->m_Device.updateDescriptorSets(
        {frameResource.m_PrimaryDescriptorSet->matrices =
             frameResource.m_StorageBuffer.m_Buffer});
}

void Renderer_VX::ReleaseFrame(unsigned frameNumber) {
    ReleaseAllResourceUse(2u << frameNumber);
    ReleaseFrameResource(m_FrameResources[frameNumber]);
}

Rml::CompiledGeometryHandle
Renderer_VX::CompileGeometry(Rml::Span<const Rml::Vertex> vertices,
                             Rml::Span<const int> indices) {
    return ~m_GeometryResources.Create(CreateGeometry(vertices, indices));
}

void Renderer_VX::RenderGeometry(Rml::CompiledGeometryHandle geometry,
                                 Rml::Vector2f translation,
                                 Rml::TextureHandle texture) {
    const uint8_t useFlag = 2u << m_Gfx->m_FrameIndex;
    const auto& g = m_GeometryResources.Use(~geometry, useFlag);
    auto pipeline = m_ColorPipeline;
    auto pipelineLayout = m_PrimaryPipelineLayout;

    ActivateLayerRendering();

    if (texture) {
        pipeline = m_TexturePipeline;
        pipelineLayout = m_TexturePipelineLayout;
        const auto& t = m_TextureResources.Use(~texture, useFlag);
        const vx::DescriptorSet<TextureDescriptorSet> descriptorSet;
        m_CommandBuffer.cmdPushDescriptorSetKHR(
            vk::PipelineBindPoint::eGraphics, pipelineLayout, 1,
            {descriptorSet->tex = t.m_ImageView});
    }
    m_CommandBuffer.cmdBindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

    m_CommandBuffer.cmdPushConstant(pipelineLayout,
                                    vk::ShaderStageFlagBits::bVertex,
                                    VX_FIELD(VsInput, translate) = translation);
    g.Draw(m_CommandBuffer);
}

void Renderer_VX::ReleaseAllResourceUse(uint8_t useFlags) {
    m_GeometryResources.ReleaseAllUse(*this, useFlags);
    m_TextureResources.ReleaseAllUse(*this, useFlags);
    m_BufferResources.ReleaseAllUse(*this, useFlags);
}

void Renderer_VX::DestroyResource(BufferResource& b) {
    const auto allocator = m_Gfx->m_Allocator;
    allocator.destroyBuffer(b.m_Buffer, b.m_Allocation);
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
    const auto transformIdx =
        transform ? CreateMatrix(Project(m_Gfx->m_FrameExtent, *transform)) : 0;
    m_CommandBuffer.cmdPushConstant(
        m_PrimaryPipelineLayout, vk::ShaderStageFlagBits::bVertex,
        VX_FIELD(VsInput, transformIdx) = transformIdx);
}

void Renderer_VX::EnableClipMask(bool enable) {
    m_EnableClipMask = enable;
    m_CommandBuffer.cmdSetStencilTestEnable(m_EnableClipMask);
}

void Renderer_VX::RenderToClipMask(Rml::ClipMaskOperation operation,
                                   Rml::CompiledGeometryHandle geometry,
                                   Rml::Vector2f translation) {
    const uint8_t useFlag = 2u << m_Gfx->m_FrameIndex;
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
    m_CommandBuffer.cmdPushConstant(m_PrimaryPipelineLayout,
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

struct GradientData {
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
        return offsetof(GradientData, stopColors) +
               sizeof(Rml::Colourf) * numStops;
    }
};

struct Renderer_VX::ShaderBase {
    enum Type { Gradient, Creation };

    Type type;
    constexpr ShaderBase(Type type) : type(type) {}
};

struct Renderer_VX::GradientShader : ShaderBase {
    GradientShader() : ShaderBase(Gradient) {}
    uintptr_t uniformBuffer;

    void Destroy(Renderer_VX* p) {
        p->m_BufferResources.Release(*p, uniformBuffer);
        delete this;
    }
};

struct Renderer_VX::CreationShader : ShaderBase {
    CreationShader() : ShaderBase(Creation) {}
    Rml::Vector2f dimensions;

    void Destroy(Renderer_VX*) { delete this; }
};

template<class T>
inline Rml::CompiledShaderHandle CreateShader(T&& shader) {
    return reinterpret_cast<Rml::CompiledShaderHandle>(
        new T(std::move(shader)));
}

Rml::CompiledShaderHandle
Renderer_VX::CompileShader(const Rml::String& name,
                           const Rml::Dictionary& parameters) {
    GradientData gradientData;
    bool invalid = false;
    if (name == "linear-gradient") {
        gradientData.func = GradientData::LINEAR << 1;
        gradientData.pos = Rml::Get(parameters, "p0", Rml::Vector2f(0.f));
        gradientData.vec =
            Rml::Get(parameters, "p1", Rml::Vector2f(0.f)) - gradientData.pos;
    } else if (name == "radial-gradient") {
        gradientData.func = GradientData::RADIAL << 1;
        gradientData.pos = Rml::Get(parameters, "center", Rml::Vector2f(0.f));
        gradientData.vec = Rml::Vector2f(1.f) /
                           Rml::Get(parameters, "radius", Rml::Vector2f(1.f));
    } else if (name == "conic-gradient") {
        gradientData.func = GradientData::CONIC << 1;
        gradientData.pos = Rml::Get(parameters, "center", Rml::Vector2f(0.f));
        const float angle = Rml::Get(parameters, "angle", 0.f);
        gradientData.vec = {Rml::Math::Cos(angle), Rml::Math::Sin(angle)};
    } else if (name == "shader") {
        const Rml::String value = Rml::Get(parameters, "value", Rml::String());
        if (value == "creation") {
            CreationShader shader;
            shader.dimensions =
                Rml::Get(parameters, "dimensions", Rml::Vector2f(0.f));
            return CreateShader(std::move(shader));
        }
        invalid = true;
    } else {
        invalid = true;
    }

    if (invalid) {
        Rml::Log::Message(Rml::Log::LT_WARNING, "Unsupported shader type '%s'.",
                          name.c_str());
        return {};
    }

    if (Rml::Get(parameters, "repeating", false))
        gradientData.func |= GradientData::REPEATING;
    gradientData.ApplyColorStopList(parameters);

    const auto size = gradientData.GetUsedSize();
    vma::AllocationInfo allocInfo;
    const auto b = CreateBufferResource(
        size, vk::BufferUsageFlagBits::bUniformBuffer, &allocInfo);
    std::memcpy(allocInfo.getMappedData(), &gradientData, size);

    GradientShader gradientShader;
    gradientShader.uniformBuffer = m_BufferResources.Create(b);
    return CreateShader(std::move(gradientShader));
}

void Renderer_VX::RenderShader(Rml::CompiledShaderHandle shader,
                               Rml::CompiledGeometryHandle geometry,
                               Rml::Vector2f translation,
                               Rml::TextureHandle /*texture*/) {
    const uint8_t useFlag = 2u << m_Gfx->m_FrameIndex;
    const auto& g = m_GeometryResources.Use(~geometry, useFlag);

    ActivateLayerRendering();

    m_CommandBuffer.cmdPushConstant(m_PrimaryPipelineLayout,
                                    vk::ShaderStageFlagBits::bVertex,
                                    VX_FIELD(VsInput, translate) = translation);

    VisitShader(reinterpret_cast<ShaderBase*>(shader),
                [this](auto* p) { SetShader(*p); });

    g.Draw(m_CommandBuffer);
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
        const auto& frameResource = m_FrameResources[m_Gfx->m_FrameIndex];
        m_Gfx->m_Device.updateDescriptorSets(
            {frameResource.m_PrimaryDescriptorSet->textures[index] =
                 fb.m_ImageView});
    }
    return fb;
}

vk::Image Renderer_VX::BeginPostprocess(unsigned index, bool load) {
    const auto& colorImage = GetPostprocess(index);
    vx::ImageMemoryBarrierState colorImageBarrier;

    colorImageBarrier.init(
        colorImage.m_Image,
        vx::subresourceRange(vk::ImageAspectFlagBits::bColor));
    if (load) {
        colorImageBarrier.setOldLayout(vk::ImageLayout::eTransferDstOptimal);
        colorImageBarrier.setSrcStageAccess(
            vk::PipelineStageFlagBits2::bTransfer,
            vk::AccessFlagBits2::bTransferWrite);
    } else {
        colorImageBarrier.setSrcStageAccess(
            vk::PipelineStageFlagBits2::bFragmentShader,
            vk::AccessFlagBits2::bShaderRead);
    }
    colorImageBarrier.setNewLayout(vk::ImageLayout::eAttachmentOptimal);
    colorImageBarrier.setDstStageAccess(
        vk::PipelineStageFlagBits2::bColorAttachmentOutput,
        vk::AccessFlagBits2::bColorAttachmentWrite);

    m_CommandBuffer.cmdPipelineBarriers(colorImageBarrier);

    vk::RenderingAttachmentInfo colorAttachmentInfo;
    colorAttachmentInfo.setImageView(colorImage.m_ImageView);
    colorAttachmentInfo.setImageLayout(colorImageBarrier.getNewLayout());
    colorAttachmentInfo.setStoreOp(vk::AttachmentStoreOp::eStore);
    if (load) {
        colorAttachmentInfo.setLoadOp(vk::AttachmentLoadOp::eLoad);
    } else {
        colorAttachmentInfo.setLoadOp(vk::AttachmentLoadOp::eClear);
        colorAttachmentInfo.setClearValue({.color = vk::ClearColorValue()});
    }

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

void Renderer_VX::SetPostprocessSample(unsigned index) {
    m_CommandBuffer.cmdPushConstant(m_PrimaryPipelineLayout,
                                    vk::ShaderStageFlagBits::bFragment,
                                    VX_FIELD(FsInput, texIdx) = index);
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

void Renderer_VX::RenderPassthrough(vk::Pipeline pipeline,
                                    vk::Bool32 colorBlendEnable) {
    m_CommandBuffer.cmdBindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

    m_CommandBuffer.cmdSetColorBlendEnableEXT(0, 1, &colorBlendEnable);
    if (colorBlendEnable) {
        const auto colorBlendEquation =
            MakeColorBlendEquation(vk::BlendOp::eAdd, vk::BlendFactor::eOne,
                                   vk::BlendFactor::eOneMinusSrcAlpha);
        m_CommandBuffer.cmdSetColorBlendEquationEXT(0, 1, &colorBlendEquation);
    }

    m_GeometryResources.Get(~m_FullscreenQuadGeometry).Draw(m_CommandBuffer);
}

Rml::CompiledGeometryHandle
Renderer_VX::UseFullscreenQuad(Rml::Vector2f uv_offset,
                               Rml::Vector2f uv_scaling) {
    if (uv_offset == Rml::Vector2f() && uv_scaling == Rml::Vector2f(1.f)) {
        return m_FullscreenQuadGeometry;
    }
    const QuadMesh mesh(Rml::Vector2f(-1), Rml::Vector2f(2), uv_offset,
                        uv_scaling);

    const uint8_t useFlag = 2u << m_Gfx->m_FrameIndex;
    return ~m_GeometryResources.Create(
        CreateGeometry(mesh.vertices, mesh.indices), useFlag);
}

void Renderer_VX::ReleaseShader(Rml::CompiledShaderHandle shader) {
    RMLUI_ASSERT(shader);
    VisitShader(reinterpret_cast<ShaderBase*>(shader),
                [this](auto* p) { p->Destroy(this); });
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
    const auto postprocessImage = GetPostprocess(m_PostprocessIndex).m_Image;
    ResolveLayer(source, postprocessImage);

    TransitionToSample(postprocessImage, true);

    // Render the filters, the Postprocess(0) framebuffer is used for both
    // input and output.
    RenderFilters(filters);

    BeginLayerRendering(destination);
    ActivateLayerRendering();
    SetPostprocessSample(m_PostprocessIndex);

    RenderPassthrough(m_Gfx->m_SampleCount == vk::SampleCountFlagBits::b1
                          ? m_PassthroughPipeline
                          : m_MsPassthroughPipeline,
                      blend_mode == BlendMode::Blend);

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
    const auto postprocessImage = GetPostprocess(m_PostprocessIndex).m_Image;
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

struct Renderer_VX::FilterBase {
    enum Type { Passthrough, Blur, DropShadow, ColorMatrix, MaskImage };

    Type type;
    constexpr FilterBase(Type type) : type(type) {}
};

struct Renderer_VX::PassthroughFilter : FilterBase {
    constexpr PassthroughFilter() : FilterBase(Passthrough) {}
    float blendFactor;
};

struct Renderer_VX::BlurFilter : FilterBase {
    constexpr BlurFilter() : FilterBase(Blur) {}
    float sigma;
};

struct Renderer_VX::DropShadowFilter : FilterBase {
    DropShadowFilter() noexcept : FilterBase(DropShadow) {}
    float sigma;
    Rml::Vector2f offset;
    Rml::ColourbPremultiplied color;
};

struct Renderer_VX::ColorMatrixFilter : FilterBase {
    ColorMatrixFilter() noexcept : FilterBase(ColorMatrix) {}
    Rml::Matrix4f colorMatrix;
};

struct Renderer_VX::MaskImageFilter : FilterBase {
    constexpr MaskImageFilter() : FilterBase(MaskImage) {}
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
        filter.blendFactor = Rml::Get(parameters, "value", 1.0f);
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
void Renderer_VX::VisitShader(ShaderBase* p, F f) {
    switch (p->type) {
    case ShaderBase::Gradient: return f(static_cast<GradientShader*>(p));
    case ShaderBase::Creation: return f(static_cast<CreationShader*>(p));
    }
}

template<class F>
void Renderer_VX::VisitFilter(FilterBase* p, F f) {
    switch (p->type) {
    case FilterBase::Passthrough: return f(static_cast<PassthroughFilter*>(p));
    case FilterBase::Blur: return f(static_cast<BlurFilter*>(p));
    case FilterBase::DropShadow: return f(static_cast<DropShadowFilter*>(p));
    case FilterBase::ColorMatrix: return f(static_cast<ColorMatrixFilter*>(p));
    case FilterBase::MaskImage: return f(static_cast<MaskImageFilter*>(p));
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
        &m_PrimaryDescriptorSetLayout, context,
        vk::DescriptorSetLayoutCreateFlagBits::bUpdateAfterBindPool));

    vk::PushConstantRange pushConstantRanges[2];
    auto& [vertPushConstantRange, fragPushConstantRange] = pushConstantRanges;
    vertPushConstantRange.setStageFlags(vk::ShaderStageFlagBits::bVertex);
    VsInput::SetRange(vertPushConstantRange);

    fragPushConstantRange.setStageFlags(vk::ShaderStageFlagBits::bFragment);
    FsInput::SetRange(fragPushConstantRange);

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    pipelineLayoutInfo.setPushConstantRanges(pushConstantRanges);
    pipelineLayoutInfo.setPushConstantRangeCount(2);

    vk::DescriptorSetLayout descriptorSetLayouts[] = {
        m_PrimaryDescriptorSetLayout, m_TextureDescriptorSetLayout};
    pipelineLayoutInfo.setSetLayouts(descriptorSetLayouts);

    pipelineLayoutInfo.setSetLayoutCount(1);

    m_PrimaryPipelineLayout =
        device.createPipelineLayout(pipelineLayoutInfo).get();

    pipelineLayoutInfo.setSetLayoutCount(2);

    m_TexturePipelineLayout =
        device.createPipelineLayout(pipelineLayoutInfo).get();

    descriptorSetLayouts[1] = m_UniformDescriptorSetLayout;

    m_GradientPipelineLayout =
        device.createPipelineLayout(pipelineLayoutInfo).get();

#ifndef NDEBUG
    device.setDebugUtilsObjectNameEXT(m_PrimaryPipelineLayout,
                                      "m_PrimaryPipelineLayout");
    device.setDebugUtilsObjectNameEXT(m_TexturePipelineLayout,
                                      "m_TexturePipelineLayout");
    device.setDebugUtilsObjectNameEXT(m_GradientPipelineLayout,
                                      "m_GradientPipelineLayout");
#endif // !NDEBUG
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
    const auto dropShadowFragShader =
        device.createShaderModule(shader_frag_drop_shadow).get();
    const auto creationFragShader =
        device.createShaderModule(shader_frag_creation).get();

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

    pipelineBuilder.setLayout(m_PrimaryPipelineLayout);
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

    shaderStageInfos[1].setModule(creationFragShader);

    m_CreationPipeline = pipelineBuilder.build(device).get();

    pipelineBuilder.setLayout(m_GradientPipelineLayout);
    shaderStageInfos[1].setModule(gradientFragShader);
    m_GradientPipeline = pipelineBuilder.build(device).get();

    pipelineBuilder.setLayout(m_TexturePipelineLayout);
    shaderStageInfos[1].setModule(textureFragShader);

    m_TexturePipeline = pipelineBuilder.build(device).get();

    pipelineBuilder.setLayout(m_PrimaryPipelineLayout);

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

    shaderStageInfos[1].setModule(dropShadowFragShader);

    m_DropShadowPipeline = pipelineBuilder.build(device).get();

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
    device.destroyShaderModule(dropShadowFragShader);
    device.destroyShaderModule(creationFragShader);

#ifndef NDEBUG
    device.setDebugUtilsObjectNameEXT(m_ClipPipeline, "m_ClipPipeline");
    device.setDebugUtilsObjectNameEXT(m_ColorPipeline, "m_ColorPipeline");
    device.setDebugUtilsObjectNameEXT(m_GradientPipeline, "m_GradientPipeline");
    device.setDebugUtilsObjectNameEXT(m_TexturePipeline, "m_TexturePipeline");
    device.setDebugUtilsObjectNameEXT(m_PassthroughPipeline,
                                      "m_PassthroughPipeline");
    device.setDebugUtilsObjectNameEXT(m_ColorMatrixPipeline,
                                      "m_ColorMatrixPipeline");
    device.setDebugUtilsObjectNameEXT(m_BlendMaskPipeline,
                                      "m_BlendMaskPipeline");
    device.setDebugUtilsObjectNameEXT(m_BlurPipeline, "m_BlurPipeline");
    device.setDebugUtilsObjectNameEXT(m_DropShadowPipeline,
                                      "m_DropShadowPipeline");
    device.setDebugUtilsObjectNameEXT(m_CreationPipeline, "m_CreationPipeline");
#endif
}

Renderer_VX::BufferResource
Renderer_VX::CreateBufferResource(size_t size, vk::BufferUsageFlags usageFlags,
                                  vma::AllocationInfo* allocInfo) {
    BufferResource b;

    vk::BufferCreateInfo bufferInfo;
    bufferInfo.setSize(size);
    bufferInfo.setUsage(usageFlags);
    bufferInfo.setSharingMode(vk::SharingMode::eExclusive);

    vma::AllocationCreateInfo allocationInfo;
    allocationInfo.setFlags(
        vma::AllocationCreateFlagBits::bMapped |
        vma::AllocationCreateFlagBits::bHostAccessSequentialWrite);
    allocationInfo.setUsage(vma::MemoryUsage::eAutoPreferDevice);

    b.m_Buffer = m_Gfx->m_Allocator
                     .createBuffer(bufferInfo, allocationInfo, &b.m_Allocation,
                                   allocInfo)
                     .get();

    return b;
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

    vma::AllocationInfo allocInfo;
    static_cast<BufferResource&>(g) =
        CreateBufferResource(vertexBytes + indexBytes,
                             vk::BufferUsageFlagBits::bVertexBuffer |
                                 vk::BufferUsageFlagBits::bIndexBuffer,
                             &allocInfo);

    auto p = static_cast<uint8_t*>(allocInfo.getMappedData());
    std::memcpy(p, vertices.data(), vertexBytes);
    p += vertexBytes;
    std::memcpy(p, indices.data(), indexBytes);

    return g;
}

void Renderer_VX::ReleaseFrameResource(FrameResource& frameResource) {
    if (frameResource.m_StorageBuffer.m_Buffer) {
        DestroyResource(frameResource.m_StorageBuffer);
    }
}

void Renderer_VX::SetShader(const GradientShader& shader) {
    const uint8_t useFlag = 2u << m_Gfx->m_FrameIndex;

    m_CommandBuffer.cmdBindPipeline(vk::PipelineBindPoint::eGraphics,
                                    m_GradientPipeline);

    const vx::DescriptorSet<UniformDescriptorSet> descriptorSet;
    m_CommandBuffer.cmdPushDescriptorSetKHR(
        vk::PipelineBindPoint::eGraphics, m_GradientPipelineLayout, 1,
        {descriptorSet->uniform =
             m_BufferResources.Use(shader.uniformBuffer, useFlag).m_Buffer});
}

void Renderer_VX::SetShader(const CreationShader& shader) {
    m_CommandBuffer.cmdBindPipeline(vk::PipelineBindPoint::eGraphics,
                                    m_CreationPipeline);

    CreationParams creationParams;
    creationParams.value = float(Rml::GetSystemInterface()->GetElapsedTime());
    creationParams.dimensions = shader.dimensions;

    m_CommandBuffer.cmdPushConstants(m_PrimaryPipelineLayout,
                                     vk::ShaderStageFlagBits::bFragment, 12,
                                     sizeof(CreationParams), &creationParams);
}

void Renderer_VX::RenderFilters(
    Rml::Span<const Rml::CompiledFilterHandle> filterHandles) {
    for (const auto filterHandle : filterHandles) {
        VisitFilter(reinterpret_cast<FilterBase*>(filterHandle),
                    [this](auto* p) { RenderFilter(*p); });
    }
}

void Renderer_VX::RenderFilter(const PassthroughFilter& filter) {
    const auto postprocesImage = BeginPostprocess(m_PostprocessIndex ^ 1);
    SetPostprocessSample(m_PostprocessIndex);

    m_CommandBuffer.cmdBindPipeline(vk::PipelineBindPoint::eGraphics,
                                    m_PassthroughPipeline);

    const vk::Bool32 colorBlendEnable = true;
    m_CommandBuffer.cmdSetColorBlendEnableEXT(0, 1, &colorBlendEnable);
    const auto colorBlendEquation = MakeColorBlendEquation(
        vk::BlendOp::eAdd, vk::BlendFactor::eConstantColor,
        vk::BlendFactor::eZero);
    m_CommandBuffer.cmdSetColorBlendEquationEXT(0, 1, &colorBlendEquation);
    const float blendConstants[4] = {filter.blendFactor, filter.blendFactor,
                                     filter.blendFactor, filter.blendFactor};
    m_CommandBuffer.cmdSetBlendConstants(blendConstants);

    m_GeometryResources.Get(~m_FullscreenQuadGeometry).Draw(m_CommandBuffer);
    m_CommandBuffer.cmdEndRendering();
    TransitionToSample(postprocesImage, false);
    m_PostprocessIndex ^= 1;
}

void Renderer_VX::RenderFilter(const BlurFilter& filter) {
    RenderBlur(filter.sigma, {m_PostprocessIndex, m_PostprocessIndex ^ 1});
    TransitionToSample(GetPostprocess(m_PostprocessIndex).m_Image, true);
}

void Renderer_VX::RenderFilter(const DropShadowFilter& filter) {
    const auto postprocesImage = BeginPostprocess(m_PostprocessIndex ^ 1);
    SetPostprocessSample(m_PostprocessIndex);

    m_CommandBuffer.cmdBindPipeline(vk::PipelineBindPoint::eGraphics,
                                    m_DropShadowPipeline);

    const auto extent = m_Gfx->m_FrameExtent;
    DropShadowParams dropShadowParams;
    dropShadowParams.color = ConvertToColorf(filter.color);
    dropShadowParams.SetTexCoordLimits(m_Scissor, extent);

    m_CommandBuffer.cmdPushConstant(
        m_PrimaryPipelineLayout, vk::ShaderStageFlagBits::bFragment,
        VX_FIELD(FsInput, dropShadowParams) = dropShadowParams);

    const auto uv_offset = -filter.offset / Rml::Vector2f(float(extent.width),
                                                          float(extent.height));
    const auto quadGeometry = UseFullscreenQuad(uv_offset, Rml::Vector2f(1.f));
    m_GeometryResources.Get(~quadGeometry).Draw(m_CommandBuffer);

    if (filter.sigma >= 0.5f) {
        m_CommandBuffer.cmdEndRendering();
        RenderBlur(filter.sigma, {m_PostprocessIndex ^ 1, 2});
        BeginPostprocess(m_PostprocessIndex ^ 1, true);
        SetPostprocessSample(m_PostprocessIndex);
    }

    RenderPassthrough(m_PassthroughPipeline, true);
    m_CommandBuffer.cmdEndRendering();
    TransitionToSample(postprocesImage, false);
    m_PostprocessIndex ^= 1;
}

void Renderer_VX::RenderFilter(const ColorMatrixFilter& filter) {
    const auto postprocesImage = BeginPostprocess(m_PostprocessIndex ^ 1);
    SetPostprocessSample(m_PostprocessIndex);

    m_CommandBuffer.cmdBindPipeline(vk::PipelineBindPoint::eGraphics,
                                    m_ColorMatrixPipeline);

    m_CommandBuffer.cmdPushConstant(
        m_PrimaryPipelineLayout, vk::ShaderStageFlagBits::bFragment,
        VX_FIELD(FsInput, colorMatrixIdx) = CreateMatrix(filter.colorMatrix));

    m_GeometryResources.Get(~m_FullscreenQuadGeometry).Draw(m_CommandBuffer);
    m_CommandBuffer.cmdEndRendering();
    TransitionToSample(postprocesImage, false);
    m_PostprocessIndex ^= 1;
}

void Renderer_VX::RenderFilter(const MaskImageFilter&) {
    const auto postprocesImage = BeginPostprocess(m_PostprocessIndex ^ 1);
    SetPostprocessSample(m_PostprocessIndex);

    m_CommandBuffer.cmdBindPipeline(vk::PipelineBindPoint::eGraphics,
                                    m_BlendMaskPipeline);

    m_GeometryResources.Get(~m_FullscreenQuadGeometry).Draw(m_CommandBuffer);
    m_CommandBuffer.cmdEndRendering();
    TransitionToSample(postprocesImage, false);
    m_PostprocessIndex ^= 1;
}

void Renderer_VX::RenderBlur(float sigma, const unsigned (&postprocess)[2]) {
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
        SetPostprocessSample(postprocess[j]);
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
        SetPostprocessSample(postprocess[0]);
        m_GeometryResources.Get(~m_FullscreenQuadGeometry)
            .Draw(m_CommandBuffer);
        m_CommandBuffer.cmdEndRendering();
        TransitionToSample(postprocesImage, false);
    }

    m_CommandBuffer.cmdBindPipeline(vk::PipelineBindPoint::eGraphics,
                                    m_BlurPipeline);

    // Set up uniforms.
    BlurParams blurParams;
    blurParams.SetBlurWeights(sigma);
    blurParams.SetTexCoordLimits(scissor, extent);

    m_CommandBuffer.cmdPushConstant(m_PrimaryPipelineLayout,
                                    vk::ShaderStageFlagBits::bFragment,
                                    VX_FIELD(FsInput, blurParams) = blurParams);

    // Blur render pass - vertical.
    const auto postprocesImage0 = BeginPostprocess(postprocess[0]);
    SetPostprocessSample(postprocess[1]);

    m_CommandBuffer.cmdPushConstant(
        m_PrimaryPipelineLayout, vk::ShaderStageFlagBits::bVertex,
        VX_FIELD(BlurVsInput, texelOffset) = {0.f, 1.f / extent.height});
    m_GeometryResources.Get(~m_FullscreenQuadGeometry).Draw(m_CommandBuffer);
    m_CommandBuffer.cmdEndRendering();
    TransitionToSample(postprocesImage0, false);

    // Blur render pass - horizontal.
    const auto postprocesImage1 = BeginPostprocess(postprocess[1]);
    SetPostprocessSample(postprocess[0]);

    m_CommandBuffer.cmdPushConstant(
        m_PrimaryPipelineLayout, vk::ShaderStageFlagBits::bVertex,
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
}
