#pragma once

#include <RmlUi/Core/RenderInterface.h>
#include <vx.hpp>

struct Renderer_VX : Rml::RenderInterface {
    struct Backend {
        vx::Device (&GetDevice)(Renderer_VX*);
        vma::Allocator (&GetAllocator)(Renderer_VX*);
        vk::RenderPass (&GetRenderPass)(Renderer_VX*);
        vk::Extent2D (&GetFrameExtent)(Renderer_VX*);
        vx::CommandBuffer (&BeginCommands)(Renderer_VX*);
        void (&EndCommands)(Renderer_VX*);
    };

    Renderer_VX();
    ~Renderer_VX();

    bool Init(const Backend& backend, uint32_t frameCount);
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

private:
    struct MyDescriptorSet;
    struct FrameResources;
    enum { ColorPipeline, TexturePipeline, PipelineCount };

    void InitPipelines();

    Rml::TextureHandle CreateTexture(vk::Buffer buffer,
                                     Rml::Vector2i dimensions);

    const Backend* m_Backend = nullptr;
    std::unique_ptr<FrameResources[]> m_FrameResources;
    vx::DescriptorSetLayout<MyDescriptorSet> m_DescriptorSetLayout;
    vk::PipelineLayout m_PipelineLayouts[PipelineCount];
    vk::Pipeline m_Pipelines[PipelineCount];
    vk::Sampler m_Sampler;
    vx::CommandBuffer m_CommandBuffer;
    uint32_t m_FrameNumber = 0;
    bool m_EnableScissor = false;
    bool m_HasTransform = false;
};