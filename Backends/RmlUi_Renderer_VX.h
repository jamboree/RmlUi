#pragma once

#include <RmlUi/Core/RenderInterface.h>
#include <vx.hpp>

struct Renderer_VX : Rml::RenderInterface {
    struct Context {
        vx::Device (&GetDevice)(Renderer_VX*);
        vma::Allocator (&GetAllocator)(Renderer_VX*);
        vk::Extent2D (&GetFrameExtent)(Renderer_VX*);
        vx::CommandBuffer (&BeginTransfer)(Renderer_VX*);
        void (&EndTransfer)(Renderer_VX*);
    };

    Renderer_VX();
    ~Renderer_VX();

    bool Init(const Context& context, vk::RenderPass renderPass,
              uint32_t frameCount);
    void Shutdown();

    void BeginFrame(vx::CommandBuffer commandBuffer, uint32_t frame);
    void EndFrame();
    void ResetFrame(uint32_t frame);

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
    struct FrameResources;

    void InitPipelineLayouts();
    void InitPipelines(vk::RenderPass renderPass);

    Rml::TextureHandle CreateTexture(vk::Buffer buffer,
                                     Rml::Vector2i dimensions);

    const Context* m_Context = nullptr;
    std::unique_ptr<FrameResources[]> m_FrameResources;
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