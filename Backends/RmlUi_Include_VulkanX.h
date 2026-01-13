#pragma once

#include <volk.h>
#include <vklite/vulkan.hpp>
#include <vklite/vk_mem_alloc.hpp>
#include <array>
#include <memory>
#include <boost/pfr/core.hpp>

namespace vk = vklite;
namespace vma = vk::vma;

namespace vklite {
    inline bool operator==(const Offset2D& a, const Offset2D& b) {
        return a.x == b.x && a.y == b.y;
    }
    inline bool operator==(const Offset3D& a, const Offset3D& b) {
        return a.x == b.x && a.y == b.y && a.z == b.z;
    }
    inline bool operator==(const Extent2D& a, const Extent2D& b) {
        return a.width == b.width && a.height == b.height;
    }
    inline bool operator==(const Extent3D& a, const Extent3D& b) {
        return a.width == b.width && a.height == b.height && a.depth == b.depth;
    }
} // namespace vklite

namespace vx {
    template<class T>
    struct List {
        uint32_t count = 0;
        std::unique_ptr<T[]> data;

        T* prepare() {
            data.reset(new T[count]);
            return data.get();
        }

        T* begin() { return data.get(); }
        const T* begin() const { return data.get(); }
        T* end() { return data.get() + count; }
        const T* end() const { return data.get() + count; }
        T& operator[](size_t i) noexcept { return data[i]; }
        const T& operator[](size_t i) const noexcept { return data[i]; }
    };

    template<class T>
    struct Ins : std::span<const T> {
        using std::span<const T>::span;

        Ins(const T& single) noexcept : std::span<const T>(&single, 1) {}
        Ins(std::initializer_list<T> list) noexcept
            : std::span<const T>(list.begin(), list.size()) {}
    };

    template<class... T>
    struct CheckDuplicate : T... {};

    template<class... T>
    constexpr void checkDuplicate(T... t) {
        CheckDuplicate{t...};
    }

    template<class T>
    inline bool extract(vk::Ret<T>& in, T& out) {
        if (in.result != vk::Result::eSuccess)
            return false;
        out = std::move(in.value);
        return true;
    }

    template<class T>
    constexpr T alignUp(T size, T align) {
        const T mask = align - 1;
        return (size + mask) & ~mask;
    }

    struct SizeAllocator {
        std::size_t offset = 0;

        constexpr std::size_t allocate(std::size_t size, std::size_t align) {
            const auto p = alignUp(offset, align);
            offset = p + size;
            return p;
        }
    };

    inline vk::Offset3D toOffset3D(vk::Offset2D offset) {
        return {offset.x, offset.y, 0};
    }

    inline vk::Extent3D toExtent3D(vk::Extent2D offset) {
        return {offset.width, offset.height, 1};
    }

    struct Range3D {
        vk::Offset3D min;
        vk::Offset3D max;

        Range3D() = default;

        Range3D(const vk::Extent3D& extent) noexcept
            : max(extent.width, extent.height, extent.depth) {}

        Range3D(const vk::Offset3D& min, const vk::Offset3D& max) noexcept
            : min(min), max(max) {}

        Range3D(const vk::Offset3D& min, const vk::Extent3D& extent) noexcept
            : min(min), max(min.x + extent.width, min.y + extent.height,
                            min.z + extent.depth) {}

        vk::Extent3D getExtent() const {
            return vk::Extent3D(max.x - min.x, max.y - min.y, max.z - min.z);
        }

        bool operator==(const Range3D&) const = default;
    };

    struct BufferOffset {
        vk::Buffer m_buffer;
        vk::DeviceSize m_offset = 0;

        BufferOffset() = default;

        BufferOffset(vk::Buffer buffer) noexcept : m_buffer(buffer) {}

        BufferOffset(vk::Buffer buffer, vk::DeviceSize offset) noexcept
            : m_buffer(buffer), m_offset(offset) {}
    };

#if VK_EXT_debug_utils
    typedef VkBool32 VKAPI_CALL DebugUtilsMessengerCallbackEXT(
        vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        vk::DebugUtilsMessageTypeFlagsEXT messageTypes,
        const vk::DebugUtilsMessengerCallbackDataEXT& callbackData,
        void* pUserData);

    struct DebugUtilsMessengerCreateInfoEXT
        : vk::DebugUtilsMessengerCreateInfoEXT {
        void setPfnUserCallback(DebugUtilsMessengerCallbackEXT* pfn) {
            pfnUserCallback =
                reinterpret_cast<PFN_vkDebugUtilsMessengerCallbackEXT>(pfn);
        }
    };
#endif

    inline std::span<const vk::Offset3D, 2> toSpan(const Range3D& range) {
        return std::span<const vk::Offset3D, 2>(&range.min, 2);
    }

    inline vk::ImageSubresourceRange
    subresourceRange(vk::ImageAspectFlags aspectFlags,
                     uint32_t layerCount = VK_REMAINING_ARRAY_LAYERS,
                     uint32_t baseArrayLayer = 0) {
        return {/*aspectMask*/ aspectFlags, /*baseMipLevel*/ 0,
                /*levelCount*/ VK_REMAINING_MIP_LEVELS, baseArrayLayer,
                layerCount};
    }

    inline vk::ImageSubresourceLayers
    subresourceLayers(vk::ImageAspectFlags aspectFlags, uint32_t layerCount,
                      uint32_t baseArrayLayer = 0) {
        return {aspectFlags, /*mipLevel*/ 0, baseArrayLayer, layerCount};
    }

    inline vk::ImageSubresourceRange subresourceRangeFromLayers(
        const vk::ImageSubresourceLayers& subresourceLayers) {
        return {/*aspectMask*/ subresourceLayers.getAspectMask(),
                /*baseMipLevel*/ subresourceLayers.mipLevel,
                /*levelCount*/ 1,
                /*baseArrayLayer*/ subresourceLayers.baseArrayLayer,
                /*layerCount*/ subresourceLayers.layerCount};
    }

    inline vk::ImageCreateInfo imageCreateInfo(vk::ImageType imageType,
                                               vk::Format format,
                                               vk::Extent3D extent,
                                               vk::ImageUsageFlags usage) {
        vk::ImageCreateInfo imageInfo;
        imageInfo.setImageType(imageType);
        imageInfo.setFormat(format);
        imageInfo.setExtent(extent);
        imageInfo.setMipLevels(1);
        imageInfo.setArrayLayers(1);
        imageInfo.setSamples(vk::SampleCountFlagBits::b1);
        imageInfo.setTiling(vk::ImageTiling::eOptimal);
        imageInfo.setUsage(usage);
        imageInfo.setSharingMode(vk::SharingMode::eExclusive);
        return imageInfo;
    }

    inline vk::ImageCreateInfo image1DCreateInfo(vk::Format format,
                                                 uint32_t width,
                                                 vk::ImageUsageFlags usage) {
        return imageCreateInfo(vk::ImageType::e1D, format,
                               vk::Extent3D(width, 1, 1), usage);
    }

    inline vk::ImageCreateInfo image2DCreateInfo(vk::Format format,
                                                 vk::Extent2D extent,
                                                 vk::ImageUsageFlags usage) {
        return imageCreateInfo(vk::ImageType::e2D, format, toExtent3D(extent),
                               usage);
    }

    inline vk::ImageViewCreateInfo
    imageViewCreateInfo(vk::ImageViewType viewType, vk::Image image,
                        vk::Format format,
                        vk::ImageSubresourceRange subresourceRange) {
        vk::ImageViewCreateInfo imageViewInfo;
        imageViewInfo.setImage(image);
        imageViewInfo.setViewType(viewType);
        imageViewInfo.setFormat(format);
        imageViewInfo.setSubresourceRange(subresourceRange);
        return imageViewInfo;
    }

    inline vk::Offset2D getOffset2(const vk::Rect2D& rect) {
        return {rect.offset.x + int32_t(rect.extent.width),
                rect.offset.y + int32_t(rect.extent.height)};
    }

    inline vk::PipelineShaderStageCreateInfo
    makePipelineShaderStageCreateInfo(vk::ShaderStageFlagBits stage,
                                      vk::ShaderModule shaderModule,
                                      const char* name = "main") {
        vk::PipelineShaderStageCreateInfo shaderStageInfo;
        shaderStageInfo.setStage(stage);
        shaderStageInfo.setName(name);
        shaderStageInfo.setModule(shaderModule);
        return shaderStageInfo;
    }

    inline vk::ComputePipelineCreateInfo
    makeComputePipelineCreateInfo(vk::PipelineLayout layout,
                                  vk::ShaderModule shaderModule,
                                  const char* name = "main") {
        vk::ComputePipelineCreateInfo pipelineInfo;
        pipelineInfo.setStage(makePipelineShaderStageCreateInfo(
            vk::ShaderStageFlagBits::bCompute, shaderModule, name));
        pipelineInfo.setLayout(layout);
        return pipelineInfo;
    }

    struct GraphicsPipelineBuilder : vk::GraphicsPipelineCreateInfo,
                                     vk::PipelineVertexInputStateCreateInfo,
                                     vk::PipelineInputAssemblyStateCreateInfo,
                                     vk::PipelineViewportStateCreateInfo,
                                     vk::PipelineRasterizationStateCreateInfo,
                                     vk::PipelineMultisampleStateCreateInfo,
                                     vk::PipelineDepthStencilStateCreateInfo,
                                     vk::PipelineColorBlendAttachmentState,
                                     vk::PipelineColorBlendStateCreateInfo,
                                     vk::PipelineDynamicStateCreateInfo {
        using GraphicsPipelineCreateInfo::attach;

        void setDepthBounds(float min, float max) {
            PipelineDepthStencilStateCreateInfo::setMinDepthBounds(min);
            PipelineDepthStencilStateCreateInfo::setMaxDepthBounds(max);
        }

        void setStencilOpState(const vk::StencilOpState& opState) {
            PipelineDepthStencilStateCreateInfo::setFront(opState);
            PipelineDepthStencilStateCreateInfo::setBack(opState);
        }

        void setStencilOpState(const vk::StencilOpState& frontOpState,
                               const vk::StencilOpState& backOpState) {
            PipelineDepthStencilStateCreateInfo::setFront(frontOpState);
            PipelineDepthStencilStateCreateInfo::setBack(backOpState);
        }

        void setColorBlend(vk::BlendOp op, vk::BlendFactor srcFactor,
                           vk::BlendFactor dstFactor) {
            PipelineColorBlendAttachmentState::setColorBlendOp(op);
            PipelineColorBlendAttachmentState::setSrcColorBlendFactor(
                srcFactor);
            PipelineColorBlendAttachmentState::setDstColorBlendFactor(
                dstFactor);
        }

        void setAlphaBlend(vk::BlendOp op, vk::BlendFactor srcFactor,
                           vk::BlendFactor dstFactor) {
            PipelineColorBlendAttachmentState::setAlphaBlendOp(op);
            PipelineColorBlendAttachmentState::setSrcAlphaBlendFactor(
                srcFactor);
            PipelineColorBlendAttachmentState::setDstAlphaBlendFactor(
                dstFactor);
        }

        void setBlend(vk::BlendOp op, vk::BlendFactor srcFactor,
                      vk::BlendFactor dstFactor) {
            setColorBlend(op, srcFactor, dstFactor);
            setAlphaBlend(op, srcFactor, dstFactor);
        }

        void setColorBlendLogicOpEnable(bool enable) {
            PipelineColorBlendStateCreateInfo::setLogicOpEnable(enable);
        }

        GraphicsPipelineBuilder() noexcept {
            PipelineViewportStateCreateInfo::setViewportCount(1);
            PipelineViewportStateCreateInfo::setScissorCount(1);

            PipelineRasterizationStateCreateInfo::setLineWidth(1.f);

            PipelineMultisampleStateCreateInfo::setRasterizationSamples(
                vk::SampleCountFlagBits::b1);
            PipelineMultisampleStateCreateInfo::setMinSampleShading(1.f);

            PipelineColorBlendStateCreateInfo::setAttachmentCount(1);
            PipelineColorBlendStateCreateInfo::setAttachments(this);

            GraphicsPipelineCreateInfo::setVertexInputState(this);
            GraphicsPipelineCreateInfo::setInputAssemblyState(this);
            GraphicsPipelineCreateInfo::setViewportState(this);
            GraphicsPipelineCreateInfo::setRasterizationState(this);
            GraphicsPipelineCreateInfo::setMultisampleState(this);
            GraphicsPipelineCreateInfo::setDepthStencilState(this);
            GraphicsPipelineCreateInfo::setColorBlendState(this);
            GraphicsPipelineCreateInfo::setDynamicState(this);
        }

        vk::Ret<vk::Pipeline>
        build(vk::Device device, vk::PipelineCache pipelineCache = {}) const {
            vk::Ret<vk::Pipeline> ret;
            ret.result = device.createGraphicsPipelines(pipelineCache, 1, this,
                                                        &ret.value);
            return ret;
        }
    };

    struct ImageMemoryBarrierState : vk::ImageMemoryBarrier2 {
        void init(vk::Image image,
                  const vk::ImageSubresourceRange& subresourceRange) {
            setImage(image);
            setOldLayout(vk::ImageLayout::eUndefined);
            setNewLayout(vk::ImageLayout::eUndefined);
            setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
            setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
            setSrcStageAccess(vk::PipelineStageFlagBits2::eNone,
                              vk::AccessFlagBits2::eNone);
            setDstStageAccess(vk::PipelineStageFlagBits2::eNone,
                              vk::AccessFlagBits2::eNone);
            setSubresourceRange(subresourceRange);
        }

        void setSrcStageAccess(vk::PipelineStageFlags2 stageMask,
                               vk::AccessFlags2 accessMask) {
            setSrcStageMask(stageMask);
            setSrcAccessMask(accessMask);
        }

        void setDstStageAccess(vk::PipelineStageFlags2 stageMask,
                               vk::AccessFlags2 accessMask) {
            setDstStageMask(stageMask);
            setDstAccessMask(accessMask);
        }

        void updateLayout(vk::ImageLayout layout) {
            setOldLayout(getNewLayout());
            setNewLayout(layout);
        }

        void updateStageAccess(vk::PipelineStageFlags2 stageMask,
                               vk::AccessFlags2 accessMask) {
            setSrcStageAccess(getDstStageMask(), getDstAccessMask());
            setDstStageAccess(stageMask, accessMask);
        }
    };

    inline Ins<vk::ImageMemoryBarrier2>
    rawIns(Ins<ImageMemoryBarrierState> ins) {
        return {static_cast<const vk::ImageMemoryBarrier2*>(ins.data()),
                ins.size()};
    }

    struct DependencyInfoBuilder : vk::DependencyInfo {
        std::type_identity<vk::MemoryBarrier2>
        set(Ins<vk::MemoryBarrier2> ins) {
            setMemoryBarrierCount(uint32_t(ins.size()));
            setMemoryBarriers(ins.data());
            return {};
        }

        std::type_identity<vk::BufferMemoryBarrier2>
        set(Ins<vk::BufferMemoryBarrier2> ins) {
            setBufferMemoryBarrierCount(uint32_t(ins.size()));
            setBufferMemoryBarriers(ins.data());
            return {};
        }

        std::type_identity<vk::ImageMemoryBarrier2>
        set(Ins<vk::ImageMemoryBarrier2> ins) {
            setImageMemoryBarrierCount(uint32_t(ins.size()));
            setImageMemoryBarriers(ins.data());
            return {};
        }
    };

    template<class... T>
    struct TypeList {};

    template<class T, class IdxSeq>
    struct EnumMembersImpl;

    template<class T, std::size_t... I>
    struct EnumMembersImpl<T, std::index_sequence<I...>> {
        using type = TypeList<boost::pfr::tuple_element_t<I, T>...>;
    };

    template<class T>
    using EnumMembers = EnumMembersImpl<
        T, std::make_index_sequence<boost::pfr::tuple_size_v<T>>>::type;

    template<class T>
    struct BindingTrait {
        using baseType = T;
        using paramType = const T&;
        static constexpr uint32_t getDescriptorCount(const void* /*limits*/) {
            return 1;
        }
        static std::span<const T> getSpan(paramType param) {
            return {&param, 1};
        }
    };

    template<class T, uint32_t N>
    struct BindingTrait<T[N]> {
        using baseType = T;
        using paramType = const T (&)[N];
        static constexpr uint32_t getDescriptorCount(const void* /*limits*/) {
            return N;
        }
        static std::span<const T> getSpan(paramType param) {
            return {param, N};
        }
    };

    template<class T>
    struct BindingTrait<T[]> {
        using baseType = T;
        using paramType = std::span<const T>;
        template<class Limits>
        static constexpr uint32_t getDescriptorCount(const Limits* limits) {
            return T::getMaxDescriptorCount(*limits);
        }
        static std::span<const T> getSpan(paramType param) { return param; }
    };

    template<class T>
    struct Binder {
        vk::DescriptorSet m_descriptorSet;
        uint32_t m_binding;

        vk::WriteDescriptorSet bind(std::span<const T> descriptors,
                                    uint32_t startElement) const noexcept {
            vk::WriteDescriptorSet write;
            write.setDstSet(m_descriptorSet);
            write.setDstBinding(m_binding);
            write.setDstArrayElement(startElement);
            write.setDescriptorCount(uint32_t(descriptors.size()));
            write.setDescriptorType(T::descriptorType);
            T::updateWriteDescriptorSet(write, descriptors.data());
            return write;
        }
    };

    template<class T>
    struct ArrayElementBinder {
        Binder<T> m_binder;
        uint32_t m_arrayElement;

        vk::WriteDescriptorSet operator=(const T& descriptor) const noexcept {
            return m_binder.bind({&descriptor, 1}, m_arrayElement);
        }
    };

    template<class T>
    concept HasImmutableSampler = requires { T::immutableSamplerIndex; };

    template<uint32_t Id, class T,
             vk::ShaderStageFlags Stages = vk::ShaderStageFlagBits::eAll,
             vk::DescriptorBindingFlags Flags = {}>
    struct Binding {
        using Trait = BindingTrait<T>;
        using Ty = Trait::baseType;

        static constexpr vk::DescriptorBindingFlags flags = Flags;

        template<class Limits>
        static vk::DescriptorSetLayoutBinding
        toDescriptorSetLayoutBinding(const Limits* limits) {
            vk::DescriptorSetLayoutBinding binding;
            binding.setBinding(Id);
            binding.setDescriptorType(Ty::descriptorType);
            binding.setStageFlags(Stages);
            binding.setDescriptorCount(Trait::getDescriptorCount(limits));
            if constexpr (HasImmutableSampler<T>) {
                binding.setImmutableSamplers(limits->immutableSamplers +
                                             T::immutableSamplerIndex);
            }
            return binding;
        }

        Binder<Ty> binder() const noexcept {
            return {*reinterpret_cast<const vk::DescriptorSet*>(this), Id};
        }

        vk::WriteDescriptorSet operator=(std::nullptr_t) const noexcept {
            return binder().bind({}, 0);
        }

        vk::WriteDescriptorSet
        operator=(typename Trait::paramType descriptor) const noexcept {
            return binder().bind(Trait::getSpan(descriptor), 0);
        }

        ArrayElementBinder<Ty> operator[](uint32_t arrayElement) const noexcept
            requires(std::is_array_v<T>)
        {
            return {binder(), arrayElement};
        }
    };

#define VX_BINDING(Id, T, ...)                                                 \
    [[msvc::no_unique_address]] ::vx::Binding<Id, T, __VA_ARGS__>

    template<class Bindings>
    struct DescriptorSetLayoutTrait;

    template<class... Binding>
    struct DescriptorSetLayoutTrait<TypeList<Binding...>> {
        template<class Limits>
        static std::array<vk::DescriptorSetLayoutBinding, sizeof...(Binding)>
        getBindings(const Limits* limits) {
            return {Binding::toDescriptorSetLayoutBinding(limits)...};
        }

        static consteval auto getBindingFlags() {
            if constexpr ((bool(Binding::flags) || ...)) {
                return std::array{Binding::flags...};
            } else {
                return Ins<vk::DescriptorBindingFlags>();
            }
        }
    };

    struct FieldValue {
        uint32_t offset;
        uint32_t size;
        const void* pValues;
    };

    template<class T, std::size_t Offset, std::size_t Size>
    struct Field {
        FieldValue operator=(const T& value) const noexcept {
            return {Offset, Size, &value};
        }
    };

#define VX_FIELD(s, m)                                                         \
    ::vx::Field<decltype(s::m), offsetof(s, m), sizeof(s::m)> {}

    struct CommandBuffer : vk::CommandBuffer {
        CommandBuffer() = default;

        CommandBuffer(vk::CommandBuffer base) noexcept
            : vk::CommandBuffer(base) {}

        void
        cmdDrawIndexed2(const vk::DrawIndexedIndirectCommand& command) const {
            vk::CommandBuffer::cmdDrawIndexed(
                command.indexCount, command.instanceCount, command.firstIndex,
                command.vertexOffset, command.firstInstance);
        }

        template<class... Barrier>
        void cmdPipelineBarriers(const Barrier&... barriers) const {
            DependencyInfoBuilder dependencyInfo;
            checkDuplicate(dependencyInfo.set(barriers)...);

            vk::CommandBuffer::cmdPipelineBarrier2(dependencyInfo);
        }

        void
        cmdCopyImageFromBuffer(vk::Image dstImage, const Range3D& dstRegion,
                               vk::ImageLayout dstImageLayout,
                               const vk::ImageSubresourceLayers& subresource,
                               BufferOffset srcBuffer, uint32_t rowLength = 0,
                               uint32_t imageHeight = 0) const {
            vk::BufferImageCopy bufferImageCopy;
            bufferImageCopy.setBufferOffset(srcBuffer.m_offset);
            bufferImageCopy.setImageSubresource(subresource);
            bufferImageCopy.setImageOffset(dstRegion.min);
            bufferImageCopy.setImageExtent(dstRegion.getExtent());
            bufferImageCopy.setBufferRowLength(rowLength);
            bufferImageCopy.setBufferImageHeight(imageHeight);

            vk::CommandBuffer::cmdCopyBufferToImage(srcBuffer.m_buffer,
                                                    dstImage, dstImageLayout, 1,
                                                    &bufferImageCopy);
        }

        void cmdCopyImage(vk::Image srcImage, const Range3D& srcRegion,
                          vk::ImageLayout srcLayout, vk::Image dstImage,
                          const Range3D& dstRegion, vk::ImageLayout dstLayout,
                          const vk::ImageSubresourceLayers& subresource) const {
            const auto srcExtent = srcRegion.getExtent();
            const auto dstExtent = dstRegion.getExtent();
            const vk::Extent3D extent{
                (std::min)(srcExtent.width, dstExtent.width),
                (std::min)(srcExtent.height, dstExtent.height),
                (std::min)(srcExtent.depth, dstExtent.depth)};

            vk::ImageCopy2 imageCopy;
            imageCopy.setSrcSubresource(subresource);
            imageCopy.setSrcOffset(srcRegion.min);
            imageCopy.setDstSubresource(subresource);
            imageCopy.setDstOffset(dstRegion.min);
            imageCopy.setExtent(extent);

            vk::CopyImageInfo2 copyImageInfo;
            copyImageInfo.setSrcImage(srcImage);
            copyImageInfo.setSrcImageLayout(srcLayout);
            copyImageInfo.setDstImage(dstImage);
            copyImageInfo.setDstImageLayout(dstLayout);
            copyImageInfo.setRegionCount(1);
            copyImageInfo.setRegions(&imageCopy);

            vk::CommandBuffer::cmdCopyImage2(copyImageInfo);
        }

        void cmdBlitImage(vk::Image srcImage, const Range3D& srcRegion,
                          vk::ImageLayout srcLayout, vk::Image dstImage,
                          const Range3D& dstRegion, vk::ImageLayout dstLayout,
                          const vk::ImageSubresourceLayers& subresource,
                          vk::Filter filter = vk::Filter::eLinear) const {
            vk::ImageBlit2 imageBlit;
            imageBlit.setSrcSubresource(subresource);
            imageBlit.setSrcOffsets(toSpan(srcRegion));
            imageBlit.setDstSubresource(subresource);
            imageBlit.setDstOffsets(toSpan(dstRegion));

            vk::BlitImageInfo2 blitImageInfo;
            blitImageInfo.setSrcImage(srcImage);
            blitImageInfo.setSrcImageLayout(srcLayout);
            blitImageInfo.setDstImage(dstImage);
            blitImageInfo.setDstImageLayout(dstLayout);
            blitImageInfo.setRegionCount(1);
            blitImageInfo.setRegions(&imageBlit);
            blitImageInfo.setFilter(filter);

            vk::CommandBuffer::cmdBlitImage2(blitImageInfo);
        }

        void cmdGenerateMipmaps(
            vk::Image image, const vk::Extent3D& extent, uint32_t mipLevels,
            vk::ImageAspectFlags aspectFlags = vk::ImageAspectFlagBits::bColor,
            uint32_t layerCount = 1, uint32_t baseArrayLayer = 0) const;

        void cmdPushConstant(vk::PipelineLayout layout,
                             vk::ShaderStageFlags stageFlags,
                             FieldValue field) {
            vk::CommandBuffer::cmdPushConstants(
                layout, stageFlags, field.offset, field.size, field.pValues);
        }

#if VK_KHR_push_descriptor
        void cmdPushDescriptorSetKHR(vk::PipelineBindPoint pipelineBindPoint,
                                     vk::PipelineLayout layout, uint32_t set,
                                     Ins<vk::WriteDescriptorSet> writes) {
            vk::CommandBuffer::cmdPushDescriptorSetKHR(
                pipelineBindPoint, layout, set, uint32_t(writes.size()),
                writes.data());
        }
#endif
    };

    template<class Def>
    struct DescriptorSetLayout : vk::DescriptorSetLayout {
        DescriptorSetLayout() = default;
        DescriptorSetLayout(vk::DescriptorSetLayout handle) noexcept
            : vk::DescriptorSetLayout(handle) {}
    };

    template<class Def>
    struct DescriptorSet : vk::DescriptorSet {
        DescriptorSet() = default;
        DescriptorSet(vk::DescriptorSet handle) noexcept
            : vk::DescriptorSet(handle) {}

        const Def* operator->() const noexcept {
            static_assert(sizeof(Def) == 1);
            return reinterpret_cast<const Def*>(this);
        }
    };

    struct BufferDescriptorBase : vk::DescriptorBufferInfo {
        BufferDescriptorBase(vk::Buffer buffer,
                             vk::DeviceSize range = VK_WHOLE_SIZE) noexcept {
            setBuffer(buffer);
            setRange(range);
        }

        BufferDescriptorBase(const BufferOffset& buffer,
                             vk::DeviceSize range = VK_WHOLE_SIZE) noexcept {
            setBuffer(buffer.m_buffer);
            setOffset(buffer.m_offset);
            setRange(range);
        }

        static void
        updateWriteDescriptorSet(vk::WriteDescriptorSet& write,
                                 const vk::DescriptorBufferInfo* info) {
            write.setBufferInfo(info);
        }
    };

    struct UniformBufferDescriptor : BufferDescriptorBase {
        static constexpr auto descriptorType =
            vk::DescriptorType::eUniformBuffer;

        using BufferDescriptorBase::BufferDescriptorBase;

        template<class Limits>
        static constexpr uint32_t getMaxDescriptorCount(const Limits& limits) {
            return limits.maxUniformBuffers;
        }
    };

    struct StorageBufferDescriptor : BufferDescriptorBase {
        static constexpr auto descriptorType =
            vk::DescriptorType::eStorageBuffer;

        using BufferDescriptorBase::BufferDescriptorBase;

        template<class Limits>
        static constexpr uint32_t getMaxDescriptorCount(const Limits& limits) {
            return limits.maxStorageBuffers;
        }
    };

    struct ImageDescriptorBase : vk::DescriptorImageInfo {
        static void
        updateWriteDescriptorSet(vk::WriteDescriptorSet& write,
                                 const vk::DescriptorImageInfo* info) {
            write.setImageInfo(info);
        }
    };

    struct SamplerDescriptor : ImageDescriptorBase {
        static constexpr auto descriptorType = vk::DescriptorType::eSampler;

        SamplerDescriptor(vk::Sampler sampler) noexcept { setSampler(sampler); }

        template<class Limits>
        static constexpr uint32_t getMaxDescriptorCount(const Limits& limits) {
            return limits.maxSamplers;
        }
    };

    template<unsigned I>
    struct ImmutableSamplerDescriptor : ImageDescriptorBase {
        static constexpr auto descriptorType = vk::DescriptorType::eSampler;
        static constexpr unsigned immutableSamplerIndex = I;

        ImmutableSamplerDescriptor(vk::Sampler) noexcept {}

        template<class Limits>
        static constexpr uint32_t getMaxDescriptorCount(const Limits& limits) {
            return limits.maxSamplers;
        }
    };

    struct CombinedImageSamplerDescriptor : ImageDescriptorBase {
        static constexpr auto descriptorType =
            vk::DescriptorType::eCombinedImageSampler;

        CombinedImageSamplerDescriptor(
            vk::Sampler sampler, vk::ImageView imageView,
            vk::ImageLayout imageLayout =
                vk::ImageLayout::eShaderReadOnlyOptimal) noexcept {
            setSampler(sampler);
            setImageView(imageView);
            setImageLayout(imageLayout);
        }

        template<class Limits>
        static constexpr uint32_t getMaxDescriptorCount(const Limits& limits) {
            return limits.maxCombinedImageSamplers;
        }
    };

    template<unsigned I>
    struct CombinedImageImmutableSamplerDescriptor : ImageDescriptorBase {
        static constexpr auto descriptorType =
            vk::DescriptorType::eCombinedImageSampler;
        static constexpr unsigned immutableSamplerIndex = I;

        CombinedImageImmutableSamplerDescriptor(
            vk::ImageView imageView,
            vk::ImageLayout imageLayout =
                vk::ImageLayout::eShaderReadOnlyOptimal) noexcept {
            setImageView(imageView);
            setImageLayout(imageLayout);
        }

        template<class Limits>
        static constexpr uint32_t getMaxDescriptorCount(const Limits& limits) {
            return limits.maxCombinedImageSamplers;
        }
    };

    struct ImageOnlyDescriptorBase : ImageDescriptorBase {
        ImageOnlyDescriptorBase(vk::ImageView imageView,
                                vk::ImageLayout imageLayout) noexcept {
            setImageView(imageView);
            setImageLayout(imageLayout);
        }
    };

    struct SampledImageDescriptor : ImageOnlyDescriptorBase {
        static constexpr auto descriptorType =
            vk::DescriptorType::eSampledImage;

        SampledImageDescriptor(
            vk::ImageView imageView,
            vk::ImageLayout imageLayout =
                vk::ImageLayout::eShaderReadOnlyOptimal) noexcept
            : ImageOnlyDescriptorBase(imageView, imageLayout) {}

        template<class Limits>
        static constexpr uint32_t getMaxDescriptorCount(const Limits& limits) {
            return limits.maxSampledImages;
        }
    };

    struct StorageImageDescriptor : ImageOnlyDescriptorBase {
        static constexpr auto descriptorType =
            vk::DescriptorType::eStorageImage;

        StorageImageDescriptor(vk::ImageView imageView) noexcept
            : ImageOnlyDescriptorBase(imageView, vk::ImageLayout::eGeneral) {}

        template<class Limits>
        static constexpr uint32_t getMaxDescriptorCount(const Limits& limits) {
            return limits.maxStorageImages;
        }
    };

    struct InputAttachmentDescriptor : ImageOnlyDescriptorBase {
        static constexpr auto descriptorType =
            vk::DescriptorType::eInputAttachment;

        InputAttachmentDescriptor(
            vk::ImageView imageView,
            vk::ImageLayout imageLayout =
                vk::ImageLayout::eAttachmentOptimal) noexcept
            : ImageOnlyDescriptorBase(imageView, imageLayout) {}

        template<class Limits>
        static constexpr uint32_t getMaxDescriptorCount(const Limits& limits) {
            return limits.maxInputAttachments;
        }
    };

    struct PhysicalDevice : vk::PhysicalDevice {
        PhysicalDevice() = default;
        PhysicalDevice(vk::PhysicalDevice handle) noexcept
            : vk::PhysicalDevice(handle) {}

        using vk::PhysicalDevice::enumerateDeviceExtensionProperties;

        vk::Ret<List<vk::ExtensionProperties>>
        enumerateDeviceExtensionProperties() const {
            vk::Ret<List<vk::ExtensionProperties>> ret;
            ret.result = vk::PhysicalDevice::enumerateDeviceExtensionProperties(
                nullptr, &ret.value.count);
            if (ret.result == vk::Result::eSuccess) {
                ret.result =
                    vk::PhysicalDevice::enumerateDeviceExtensionProperties(
                        nullptr, &ret.value.count, ret.value.prepare());
            }
            return ret;
        }

        using vk::PhysicalDevice::getQueueFamilyProperties;

        List<vk::QueueFamilyProperties> getQueueFamilyProperties() const {
            List<vk::QueueFamilyProperties> list;
            vk::PhysicalDevice::getQueueFamilyProperties(&list.count);
            vk::PhysicalDevice::getQueueFamilyProperties(&list.count,
                                                         list.prepare());
            return list;
        }

#if VK_KHR_surface
        using vk::PhysicalDevice::getSurfaceFormatsKHR;

        vk::Ret<List<vk::SurfaceFormatKHR>>
        getSurfaceFormatsKHR(vk::SurfaceKHR surface) const {
            vk::Ret<List<vk::SurfaceFormatKHR>> ret;
            ret.result = vk::PhysicalDevice::getSurfaceFormatsKHR(
                surface, &ret.value.count);
            if (ret.result == vk::Result::eSuccess) {
                ret.result = vk::PhysicalDevice::getSurfaceFormatsKHR(
                    surface, &ret.value.count, ret.value.prepare());
            }
            return ret;
        }

        using vk::PhysicalDevice::getSurfacePresentModesKHR;

        vk::Ret<List<vk::PresentModeKHR>>
        getSurfacePresentModesKHR(vk::SurfaceKHR surface) const {
            vk::Ret<List<vk::PresentModeKHR>> ret;
            ret.result = vk::PhysicalDevice::getSurfacePresentModesKHR(
                surface, &ret.value.count);
            if (ret.result == vk::Result::eSuccess) {
                ret.result = vk::PhysicalDevice::getSurfacePresentModesKHR(
                    surface, &ret.value.count, ret.value.prepare());
            }
            return ret;
        }
#endif

        uint32_t findQueueFamilyIndex(
            const List<vk::QueueFamilyProperties>& queueFamilyProperties,
            vk::QueueFlags flags, vk::SurfaceKHR surface = {}) const;
    };

    template<class Base, class... Extension>
    struct StructureChain : Base, Extension... {
        StructureChain() noexcept {
            (Base::attach(*static_cast<Extension*>(this)), ...);
        }
    };

    struct Instance : vk::Instance {
        Instance() = default;
        Instance(vk::Instance handle) noexcept : vk::Instance(handle) {}

        vk::Ret<List<PhysicalDevice>> enumeratePhysicalDevices() const {
            vk::Ret<List<PhysicalDevice>> ret;
            ret.result =
                vk::Instance::enumeratePhysicalDevices(&ret.value.count);
            if (ret.result == vk::Result::eSuccess) {
                ret.result = vk::Instance::enumeratePhysicalDevices(
                    &ret.value.count, ret.value.prepare());
            }
            return ret;
        }
    };

    inline vk::ShaderModuleCreateInfo
    shaderModuleInfo(std::span<const uint32_t> code) {
        vk::ShaderModuleCreateInfo info;
        info.setCodeSize(code.size_bytes());
        info.setCode(code.data());
        return info;
    }

    struct Device : vk::Device {
        Device() = default;
        Device(vk::Device handle) noexcept : vk::Device(handle) {}

#if VK_EXT_debug_utils
        using vk::Device::setDebugUtilsObjectNameEXT;

        vk::Result setDebugUtilsObjectNameEXT(vk::Object object,
                                              const char* name) const {
            vk::DebugUtilsObjectNameInfoEXT objectNameInfo;
            objectNameInfo.setObject(object);
            objectNameInfo.setObjectName(name);
            return vk::Device::setDebugUtilsObjectNameEXT(objectNameInfo);
        }
#endif

        vk::DeviceAddress getDeviceAddress(const BufferOffset& buffer) const {
            vk::BufferDeviceAddressInfo info;
            info.setBuffer(buffer.m_buffer);
            return getBufferDeviceAddress(info) + buffer.m_offset;
        }

        template<class Def>
        vk::Result createTypedDescriptorSetLayout(
            DescriptorSetLayout<Def>* out,
            vk::DescriptorSetLayoutCreateFlags flags = {}) const {
            using Trait = DescriptorSetLayoutTrait<EnumMembers<Def>>;
            const auto ret = createDescriptorSetLayout(
                flags, Trait::getBindings(static_cast<const void*>(nullptr)),
                Trait::getBindingFlags());
            *out = ret.value;
            return ret.result;
        }

        template<class Def, class Limits>
        vk::Result createTypedDescriptorSetLayoutWithContext(
            DescriptorSetLayout<Def>* out, const Limits& limits,
            vk::DescriptorSetLayoutCreateFlags flags = {}) const {
            using Trait = DescriptorSetLayoutTrait<EnumMembers<Def>>;
            const auto ret = createDescriptorSetLayout(
                flags, Trait::getBindings(&limits), Trait::getBindingFlags());
            *out = ret.value;
            return ret.result;
        }

        vk::Ret<vk::Semaphore> createBinarySemaphore() const {
            vk::SemaphoreCreateInfo semaphoreInfo;
            return vk::Device::createSemaphore(semaphoreInfo);
        }

        vk::Ret<vk::Semaphore>
        createTimelineSemaphore(uint64_t initialValue = 0) const {
            vk::SemaphoreCreateInfo semaphoreInfo;
            vk::SemaphoreTypeCreateInfo semaphoreTypeInfo;
            semaphoreTypeInfo.setSemaphoreType(vk::SemaphoreType::eTimeline);
            semaphoreTypeInfo.setInitialValue(initialValue);
            semaphoreInfo.attach(semaphoreTypeInfo);
            return vk::Device::createSemaphore(semaphoreInfo);
        }

        using vk::Device::createShaderModule;

        vk::Ret<vk::ShaderModule>
        createShaderModule(std::span<const uint32_t> code) const {
            return vk::Device::createShaderModule(shaderModuleInfo(code));
        }

#if VK_KHR_swapchain
        using vk::Device::getSwapchainImagesKHR;

        vk::Ret<List<vk::Image>>
        getSwapchainImagesKHR(vk::SwapchainKHR swapchain) const {
            vk::Ret<List<vk::Image>> ret;
            ret.result =
                vk::Device::getSwapchainImagesKHR(swapchain, &ret.value.count);
            if (ret.result == vk::Result::eSuccess) {
                ret.result = vk::Device::getSwapchainImagesKHR(
                    swapchain, &ret.value.count, ret.value.prepare());
            }
            return ret;
        }
#endif

        template<class Def>
        vk::Ret<DescriptorSet<Def>>
        allocateTypedDescriptorSet(vk::DescriptorPool descriptorPool,
                                   DescriptorSetLayout<Def> layout) const {
            vk::DescriptorSetAllocateInfo allocInfo;
            allocInfo.setDescriptorPool(descriptorPool);
            allocInfo.setDescriptorSetCount(1);
            allocInfo.setSetLayouts(&layout);

            DescriptorSet<Def> descriptorSet;
            const auto r = allocateDescriptorSets(allocInfo, &descriptorSet);
            return {r, descriptorSet};
        }

        template<class Def>
        vk::Ret<DescriptorSet<Def>> allocateTypedDescriptorSetWithVariableArray(
            vk::DescriptorPool descriptorPool, DescriptorSetLayout<Def> layout,
            uint32_t arraySize) const {
            vk::DescriptorSetAllocateInfo allocInfo;
            allocInfo.setDescriptorPool(descriptorPool);
            allocInfo.setDescriptorSetCount(1);
            allocInfo.setSetLayouts(&layout);
            vk::DescriptorSetVariableDescriptorCountAllocateInfo
                variableCountInfo;
            variableCountInfo.setDescriptorCounts(1);
            variableCountInfo.setDescriptorCounts(&arraySize);
            allocInfo.attach(variableCountInfo);

            DescriptorSet<Def> descriptorSet;
            const auto r = allocateDescriptorSets(allocInfo, &descriptorSet);
            return {r, descriptorSet};
        }

        template<class... Def>
        vk::Result allocateTypedDescriptorSets(
            vk::DescriptorPool descriptorPool,
            const std::tuple<DescriptorSetLayout<Def>...>& layouts,
            std::tuple<DescriptorSet<Def>...>& descriptorSets) const {
            vk::DescriptorSetAllocateInfo allocInfo;
            allocInfo.setDescriptorPool(descriptorPool);
            allocInfo.setDescriptorSetCount(sizeof...(Def));
            allocInfo.setSetLayouts(
                reinterpret_cast<const vk::DescriptorSetLayout*>(&layouts));

            return allocateDescriptorSets(
                allocInfo,
                reinterpret_cast<vk::DescriptorSet*>(&descriptorSets));
        }

        template<class... Def>
        vk::Result freeTypedDescriptorSets(
            vk::DescriptorPool descriptorPool,
            const std::tuple<DescriptorSet<Def>...>& descriptorSets) const {
            return freeDescriptorSets(
                descriptorPool, sizeof...(Def),
                reinterpret_cast<vk::DescriptorSet*>(&descriptorSets));
        }

        using vk::Device::updateDescriptorSets;

        void updateDescriptorSets(Ins<vk::WriteDescriptorSet> writes) const {
            vk::Device::updateDescriptorSets(uint32_t(writes.size()),
                                             writes.data());
        }

        using vk::Device::createDescriptorSetLayout;

        vk::Ret<vk::DescriptorSetLayout> createDescriptorSetLayout(
            vk::DescriptorSetLayoutCreateFlags flags,
            Ins<vk::DescriptorSetLayoutBinding> bindings,
            Ins<vk::DescriptorBindingFlags> bindingFlags = {}) const {
            vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutInfo;
            descriptorSetLayoutInfo.setFlags(flags);
            descriptorSetLayoutInfo.setBindingCount(uint32_t(bindings.size()));
            descriptorSetLayoutInfo.setBindings(bindings.data());
            vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo;
            if (!bindingFlags.empty()) {
                bindingFlagsInfo.setBindingCount(uint32_t(bindingFlags.size()));
                bindingFlagsInfo.setBindingFlags(bindingFlags.data());
                descriptorSetLayoutInfo.attach(bindingFlagsInfo);
            }
            return vk::Device::createDescriptorSetLayout(
                descriptorSetLayoutInfo);
        }

#if VK_EXT_host_image_copy
        vk::Result
        copyImageFromMemoryEXT(vk::Image image, const Range3D& range,
                               vk::ImageLayout imageLayout,
                               const vk::ImageSubresourceLayers& subresource,
                               const void* hostMemory, uint32_t rowLength = 0,
                               uint32_t imageHeight = 0) const {
            vk::CopyMemoryToImageInfo copyInfo;
            copyInfo.setDstImage(image);
            copyInfo.setDstImageLayout(imageLayout);

            vk::MemoryToImageCopy region;
            region.setHostPointer(hostMemory);
            region.setImageOffset(range.min);
            region.setImageExtent(range.getExtent());
            region.setImageSubresource(subresource);
            region.setMemoryRowLength(rowLength);
            region.setMemoryImageHeight(imageHeight);
            copyInfo.setRegionCount(1);
            copyInfo.setRegions(&region);

            return vk::Device::copyMemoryToImageEXT(copyInfo);
        }
#endif
    };

    template<class Ts, class Is>
    struct SpecializationMap;

    template<class... T, std::size_t... I>
    struct SpecializationMap<TypeList<T...>, std::index_sequence<I...>> {
        static constexpr auto count = sizeof...(T);
        static constexpr std::array<uint32_t, count> offsets = [] {
            SizeAllocator alloc;
            return std::array<uint32_t, count>{
                uint32_t(alloc.allocate(sizeof(T), alignof(T)))...};
        }();

        vk::SpecializationMapEntry entries[count];
        SpecializationMap()
            : entries{vk::SpecializationMapEntry(I, offsets[I], sizeof(T))...} {
        }
    };

    template<class T>
    struct Specialization : vk::SpecializationInfo {
        static constexpr auto count = boost::pfr::tuple_size_v<T>;

        SpecializationMap<EnumMembers<T>, std::make_index_sequence<count>> map;
        T data;

        Specialization() {
            setData(&data);
            setDataSize(sizeof(T));
            setMapEntryCount(count);
            setMapEntries(map.entries);
        }
    };

    struct PhysicalDeviceInfo {
        vk::PhysicalDeviceProperties properties;
        List<vk::QueueFamilyProperties> queueFamilyProperties;
        List<vk::ExtensionProperties> extensionProperties;

        bool init(PhysicalDevice physicalDevice);

        bool hasExtension(std::string_view name) const noexcept;
    };
} // namespace vx