#pragma once

#include <RmlUi/Core/RenderInterface.h>
#include "RmlUi_GfxContext_VX.h"

struct Renderer_VX : Rml::RenderInterface {
    Renderer_VX();
    ~Renderer_VX();

    bool Init(GfxContext_VX& gfx);
    void Destroy();

    void BeginFrame(vx::CommandBuffer commandBuffer);
    void EndFrame();
    void ResetRenderTarget();
    void ReleaseAllResourceUse(uint8_t useFlags);

    /// Called by RmlUi when it wants to compile geometry to be rendered later.
    Rml::CompiledGeometryHandle
    CompileGeometry(Rml::Span<const Rml::Vertex> vertices,
                    Rml::Span<const int> indices) override;
    /// Called by RmlUi when it wants to render geometry.
    void RenderGeometry(Rml::CompiledGeometryHandle handle,
                        Rml::Vector2f translation,
                        Rml::TextureHandle texture) override;
    /// Called by RmlUi when it wants to release geometry.
    void ReleaseGeometry(Rml::CompiledGeometryHandle geometry) override;

    /// Called by RmlUi when a texture is required by the library.
    Rml::TextureHandle LoadTexture(Rml::Vector2i& texture_dimensions,
                                   const Rml::String& source) override;
    /// Called by RmlUi when a texture is required to be generated from a
    /// sequence of pixels in memory.
    Rml::TextureHandle
    GenerateTexture(Rml::Span<const Rml::byte> source_data,
                    Rml::Vector2i source_dimensions) override;
    /// Called by RmlUi when a loaded or generated texture is no longer
    /// required.
    void ReleaseTexture(Rml::TextureHandle texture_handle) override;

    /// Called by RmlUi when it wants to enable or disable scissoring to clip
    /// content.
    void EnableScissorRegion(bool enable) override;
    /// Called by RmlUi when it wants to change the scissor region.
    void SetScissorRegion(Rml::Rectanglei region) override;

    /// Called by RmlUi when it wants the renderer to use a new transform
    /// matrix.
    void SetTransform(const Rml::Matrix4f* transform) override;

    /// Called by RmlUi when it wants to enable or disable the clip mask.
    void EnableClipMask(bool enable) override;

    /// Called by RmlUi when it wants to set or modify the contents of the clip
    /// mask.
    void RenderToClipMask(Rml::ClipMaskOperation operation,
                          Rml::CompiledGeometryHandle geometry,
                          Rml::Vector2f translation) override;

    /// Called by RmlUi when it wants to compile a new shader.
    Rml::CompiledShaderHandle
    CompileShader(const Rml::String& name,
                  const Rml::Dictionary& parameters) override;

    /// Called by RmlUi when it wants to render geometry using the given shader.
    void RenderShader(Rml::CompiledShaderHandle shader,
                      Rml::CompiledGeometryHandle geometry,
                      Rml::Vector2f translation,
                      Rml::TextureHandle texture) override;

    /// Called by RmlUi when it no longer needs a previously compiled shader.
    void ReleaseShader(Rml::CompiledShaderHandle shader) override;

    /// Called by RmlUi when it wants to push a new layer onto the render stack,
    /// setting it as the new render target.
    Rml::LayerHandle PushLayer() override;
    /// Composite two layers with the given blend mode and apply filters.
    void CompositeLayers(
        Rml::LayerHandle source, Rml::LayerHandle destination,
        Rml::BlendMode blend_mode,
        Rml::Span<const Rml::CompiledFilterHandle> filters) override;
    /// Called by RmlUi when it wants to pop the render layer stack, setting the
    /// new top layer as the render target.
    void PopLayer() override;

    /// Called by RmlUi when it wants to store the current layer as a new
    /// texture to be rendered later with geometry.
    Rml::TextureHandle SaveLayerAsTexture() override;

    /// Called by RmlUi when it wants to store the current layer as a mask
    /// image, to be applied later as a filter.
    Rml::CompiledFilterHandle SaveLayerAsMaskImage() override;

    /// Called by RmlUi when it wants to compile a new filter.
    Rml::CompiledFilterHandle
    CompileFilter(const Rml::String& name,
                  const Rml::Dictionary& parameters) override;
    /// Called by RmlUi when it no longer needs a previously compiled filter.
    void ReleaseFilter(Rml::CompiledFilterHandle filter) override;

private:
    struct TextureDescriptorSet;
    struct UniformDescriptorSet;
    struct BlendMaskDescriptorSet;
    struct BlurDescriptorSet;

    struct GeometryResource;
    struct TextureResource;
    struct ShaderResource;

    struct FilterBase;
    struct PassthroughFilter;
    struct BlurFilter;
    struct DropShadowFilter;
    struct ColorMatrixFilter;
    struct MaskImageFilter;

    struct LayerState;

    template<class T>
    struct ResourcePool {
        uintptr_t Create(const T& resource, uint8_t useFlags = 1u) {
            uintptr_t index;
            if (m_FreeHead < m_Count) {
                index = m_FreeHead;
                m_FreeHead = m_Elems[index].m_NextFree;
            } else {
                index = m_Count;
                Alloc();
                m_FreeHead = m_Count;
            }
            m_Uses[index] = useFlags;
            m_Elems[index].m_Resource = resource;
            return index;
        }

        const T& Get(uintptr_t index) const noexcept {
            return m_Elems[index].m_Resource;
        }

        const T& Use(uintptr_t index, uint8_t useFlag) const noexcept {
            m_Uses[index] |= useFlag;
            return m_Elems[index].m_Resource;
        }

        void Release(Renderer_VX& self, uintptr_t index) noexcept {
            if (!(m_Uses[index] &= ~1u)) {
                Free(self, index);
            }
        }

        void ReleaseAllUse(Renderer_VX& self, uint8_t useFlags) {
            for (uintptr_t index = 0; index != m_Count; ++index) {
                auto& use = m_Uses[index];
                if (use & useFlags) {
                    if (!(use &= ~useFlags)) {
                        Free(self, index);
                    }
                }
            }
        }

    private:
        union Elem {
            T m_Resource;
            uintptr_t m_NextFree;
        };

        void Alloc() {
            if (m_Count == m_Capacity) {
                m_Capacity += (m_Capacity / 2) | 16;
                m_Uses = static_cast<uint8_t*>(
                    std::realloc(m_Uses, sizeof(uint8_t) * m_Capacity));
                m_Elems = static_cast<Elem*>(
                    std::realloc(m_Elems, sizeof(Elem) * m_Capacity));
            }
            ++m_Count;
        }

        void Free(Renderer_VX& self, uintptr_t index) {
            auto& elem = m_Elems[index];
            self.DestroyResource(elem.m_Resource);
            elem.m_NextFree = m_FreeHead;
            m_FreeHead = index;
        }

        size_t m_Count = 0;
        size_t m_Capacity = 0;
        uintptr_t m_FreeHead = 0;
        uint8_t* m_Uses = nullptr;
        Elem* m_Elems = nullptr;
    };

    struct SurfaceManager {
        ~SurfaceManager();

        void Destroy(GfxContext_VX& gfx);

        void Invalidate();

        // Push a new layer. All references to previously retrieved layers are
        // invalidated.
        Rml::LayerHandle PushLayer(GfxContext_VX& gfx);

        // Pop the top layer. All references to previously retrieved layers are
        // invalidated.
        void PopLayer() {
            RMLUI_ASSERT(m_layers_size != 0);
            --m_layers_size;
        }

        const ImageAttachment& GetLayer(Rml::LayerHandle layer) const {
            RMLUI_ASSERT((size_t)layer < (size_t)m_layers_size);
            return m_layers[layer];
        }

        LayerState& GetLayerState(Rml::LayerHandle layer);

        Rml::LayerHandle GetTopLayerHandle() const {
            RMLUI_ASSERT(m_layers_size != 0);
            return Rml::LayerHandle(m_layers_size - 1);
        }

        const ImageAttachment& GetPostprocess(GfxContext_VX& gfx,
                                              unsigned index);

        void SwapPostprocessPrimarySecondary() {
            std::swap(m_postprocess[0], m_postprocess[1]);
        }

    private:
        // The number of active layers is manually tracked since we re-use the
        // framebuffers stored in the fb_layers stack.
        unsigned m_layers_size = 0;
        unsigned m_layers_capacity = 0;

        ImageAttachment* m_layers = nullptr;
        LayerState* m_layerStates = nullptr;
        ImageAttachment m_postprocess[4];
    };

    void InitPipelineLayouts();
    void InitPipelines(vk::PipelineRenderingCreateInfo& renderingInfo);

    Rml::TextureHandle CreateTexture(vk::Buffer buffer,
                                     Rml::Vector2i dimensions);

    GeometryResource CreateGeometry(Rml::Span<const Rml::Vertex> vertices,
                                    Rml::Span<const int> indices);

    ShaderResource CreateShaderResource(const void* data, size_t size);

    void DestroyResource(GeometryResource& g);
    void DestroyResource(TextureResource& t);
    void DestroyResource(ShaderResource& s);

    void BeginLayerRendering(Rml::LayerHandle handle);

    void ActivateLayerRendering();

    void EndLayerRendering();

    vk::Image BeginPostprocess(unsigned index);

    void TransitionToSample(vk::Image image, bool fromTransfer);

    void SetSample(unsigned index, vk::PipelineLayout pipelineLayout);

    void ResolveLayer(Rml::LayerHandle source, vk::Image dstImage);

    Rml::CompiledGeometryHandle UseFullscreenQuad(Rml::Vector2f uv_offset,
                                                  Rml::Vector2f uv_scaling);

    const ImageAttachment& GetTopLayer() const {
        return m_SurfaceManager.GetLayer(m_SurfaceManager.GetTopLayerHandle());
    }

    template<class F>
    static void VisitFilter(FilterBase* p, F f);

    void
    RenderFilters(Rml::Span<const Rml::CompiledFilterHandle> filterHandles);

    void RenderFilter(const FilterBase&) {}

    void RenderFilter(const PassthroughFilter& filter);

    void RenderFilter(const BlurFilter& filter);

    void RenderFilter(const ColorMatrixFilter& filter);

    void RenderFilter(const MaskImageFilter& filter);

    void RenderBlur(float sigma, const unsigned (&postprocess)[2]);

    GfxContext_VX* m_Gfx = nullptr;
    ResourcePool<GeometryResource> m_GeometryResources;
    ResourcePool<TextureResource> m_TextureResources;
    ResourcePool<ShaderResource> m_ShaderResources;
    SurfaceManager m_SurfaceManager;
    vx::DescriptorSetLayout<TextureDescriptorSet> m_TextureDescriptorSetLayout;
    vx::DescriptorSetLayout<UniformDescriptorSet> m_UniformDescriptorSetLayout;
    vx::DescriptorSetLayout<BlendMaskDescriptorSet>
        m_BlendMaskDescriptorSetLayout;
    vx::DescriptorSetLayout<BlurDescriptorSet> m_BlurDescriptorSetLayout;
    vk::PipelineLayout m_BasicPipelineLayout;
    vk::PipelineLayout m_TexturePipelineLayout;
    vk::PipelineLayout m_GradientPipelineLayout;
    vk::PipelineLayout m_ColorMatrixPipelineLayout;
    vk::PipelineLayout m_BlendMaskPipelineLayout;
    vk::PipelineLayout m_BlurPipelineLayout;
    vk::Pipeline m_ClipPipeline;
    vk::Pipeline m_ColorPipeline;
    vk::Pipeline m_GradientPipeline;
    vk::Pipeline m_TexturePipeline;
    vk::Pipeline m_PassthroughPipeline;
    vk::Pipeline m_MsPassthroughPipeline;
    vk::Pipeline m_ColorMatrixPipeline;
    vk::Pipeline m_BlendMaskPipeline;
    vk::Pipeline m_BlurPipeline;
    vk::Sampler m_Sampler;
    vx::CommandBuffer m_CommandBuffer;
    vk::Rect2D m_Scissor;
    // Rml::Matrix4f m_Transform;
    Rml::CompiledGeometryHandle m_FullscreenQuadGeometry = 0;
    Rml::LayerHandle m_CurrentLayer = 0;
    uint32_t m_StencilRef = 0;
    bool m_EnableScissor = false;
    bool m_EnableClipMask = false;
    bool m_LayerRendering = false;
    bool m_DepthStencilInitialized = false;
};