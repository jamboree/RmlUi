#pragma once

#include <RmlUi/Core/RenderInterface.h>
#include <vx.hpp>

struct RenderContext_VX {
    virtual vx::Device GetDevice() = 0;
    virtual vma::Allocator GetAllocator() = 0;
    virtual vk::Extent2D GetFrameExtent() = 0;
    virtual vx::CommandBuffer BeginTemp() = 0;
    virtual void EndTemp(vk::CommandBuffer commandBuffer) = 0;
};

struct Renderer_VX : Rml::RenderInterface {
    Renderer_VX();
    ~Renderer_VX();

    bool Init(RenderContext_VX& context,
              vk::PipelineRenderingCreateInfo& renderingInfo);
    void Shutdown();

    void BeginFrame(vx::CommandBuffer commandBuffer, uint32_t frame);
    void EndFrame();
    void ResetResources(uint8_t useFlags);

    /// Called by RmlUi when it wants to compile geometry it believes will be
    /// static for the forseeable future.
    Rml::CompiledGeometryHandle
    CompileGeometry(Rml::Span<const Rml::Vertex> vertices,
                    Rml::Span<const int> indices) override;
    /// Called by RmlUi when it wants to render application-compiled geometry.
    void RenderGeometry(Rml::CompiledGeometryHandle handle,
                        Rml::Vector2f translation,
                        Rml::TextureHandle texture) override;
    /// Called by RmlUi when it wants to release application-compiled geometry.
    void ReleaseGeometry(Rml::CompiledGeometryHandle geometry) override;

    /// Called by RmlUi when a texture is required by the library.
    Rml::TextureHandle LoadTexture(Rml::Vector2i& texture_dimensions,
                                   const Rml::String& source) override;
    /// Called by RmlUi when a texture is required to be built from an
    /// internally-generated sequence of pixels.
    Rml::TextureHandle
    GenerateTexture(Rml::Span<const Rml::byte> source_data,
                    Rml::Vector2i source_dimensions) override;
    /// Called by RmlUi when a loaded texture is no longer required.
    void ReleaseTexture(Rml::TextureHandle texture_handle) override;

    /// Called by RmlUi when it wants to enable or disable scissoring to clip
    /// content.
    void EnableScissorRegion(bool enable) override;
    /// Called by RmlUi when it wants to change the scissor region.
    void SetScissorRegion(Rml::Rectanglei region) override;

    /// Called by RmlUi when it wants to set the current transform matrix to a
    /// new matrix.
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

private:
    struct TextureDescriptorSet;
    struct GradientDescriptorSet;
    struct GeometryResource;
    struct TextureResource;
    struct ShaderResource;

    template<class T>
    struct ResourcePool {
        std::pair<uintptr_t, T*> Allocate() {
            uintptr_t index;
            if (m_FreeHead < m_Count) {
                index = m_FreeHead;
                m_FreeHead = m_Elems[index].m_NextFree;
            } else {
                index = m_Count;
                New();
                m_FreeHead = m_Count;
            }
            m_Uses[index] = 1u;
            return {index, new (m_Elems + index) T()};
        }

        const T* Use(uintptr_t index, uint8_t useFlag) const noexcept {
            m_Uses[index] |= useFlag;
            return &m_Elems[index].m_Resource;
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

        void New() {
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
            self.Destroy(elem.m_Resource);
            elem.m_NextFree = m_FreeHead;
            m_FreeHead = index;
        }

        size_t m_Count = 0;
        size_t m_Capacity = 0;
        uintptr_t m_FreeHead = 0;
        uint8_t* m_Uses = nullptr;
        Elem* m_Elems = nullptr;
    };

    void InitPipelineLayouts();
    void InitPipelines(vk::PipelineRenderingCreateInfo& renderingInfo);

    Rml::TextureHandle CreateTexture(vk::Buffer buffer,
                                     Rml::Vector2i dimensions);

    void Destroy(GeometryResource& g);
    void Destroy(TextureResource& t);
    void Destroy(ShaderResource& s);

    RenderContext_VX* m_Context = nullptr;
    ResourcePool<GeometryResource> m_GeometryResources;
    ResourcePool<TextureResource> m_TextureResources;
    ResourcePool<ShaderResource> m_ShaderResources;
    vx::DescriptorSetLayout<TextureDescriptorSet> m_TextureDescriptorSetLayout;
    vx::DescriptorSetLayout<GradientDescriptorSet>
        m_GradientDescriptorSetLayout;
    vk::PipelineLayout m_BasicPipelineLayout;
    vk::PipelineLayout m_TexturePipelineLayout;
    vk::PipelineLayout m_GradientPipelineLayout;
    vk::Pipeline m_ClipPipeline;
    vk::Pipeline m_ColorPipeline;
    vk::Pipeline m_TexturePipeline;
    vk::Pipeline m_GradientPipeline;
    vk::Sampler m_Sampler;
    vx::CommandBuffer m_CommandBuffer;
    vk::Rect2D m_Scissor;
    uint32_t m_FrameNumber = 0;
    uint32_t m_StencilRef = 0;
    bool m_EnableScissor = false;
};