#define VOLK_IMPLEMENTATION
#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 1

#include "RmlUi_Include_VulkanX.h"
#include <bit>
#include <algorithm>

uint32_t vx::PhysicalDevice::findQueueFamilyIndex(
    const List<vk::QueueFamilyProperties>& queueFamilyProperties,
    vk::QueueFlags flags, vk::SurfaceKHR surface) const {
    unsigned extraBitCountMin = ~0u;
    uint32_t index = ~0u;
    for (uint32_t i = 0; i != queueFamilyProperties.count; ++i) {
        if (surface) {
            const auto val = getSurfaceSupportKHR(i, surface);
            if (val.result != vk::Result::eSuccess || !val.value)
                continue;
        }
        const auto& prop = queueFamilyProperties[i];
        const auto queueFlags = prop.getQueueFlags();
        if (queueFlags.contains(flags)) {
            const unsigned extraBitCount =
                std::popcount((queueFlags ^ flags).toUnderlying());
            if (extraBitCount < extraBitCountMin) {
                extraBitCountMin = extraBitCount;
                index = i;
            }
        }
    }
    return index;
}

void vx::CommandBuffer::cmdGenerateMipmaps(
    bool hostInitialized, vk::Image image, const vk::Extent3D& extent,
    uint32_t mipLevels, vk::ImageAspectFlags aspectFlags, uint32_t layerCount,
    uint32_t baseArrayLayer) const {
    assert(mipLevels > 1);
    vk::ImageMemoryBarrier2 barrier;
    barrier.setImage(image);
    barrier.setSrcStageMask(vk::PipelineStageFlagBits2::bTransfer);
    barrier.setDstStageMask(vk::PipelineStageFlagBits2::bTransfer);
    barrier.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
    barrier.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
    barrier.setOldLayout(vk::ImageLayout::eTransferDstOptimal);
    barrier.setNewLayout(vk::ImageLayout::eTransferSrcOptimal);
    barrier.setSrcAccessMask(vk::AccessFlagBits2::bTransferWrite);
    barrier.setDstAccessMask(vk::AccessFlagBits2::bTransferRead);
    auto& subresourceRange =
        static_cast<vk::ImageSubresourceRange&>(barrier.subresourceRange);
    subresourceRange.setAspectMask(aspectFlags);
    subresourceRange.setLevelCount(1);
    subresourceRange.setLayerCount(layerCount);
    subresourceRange.setBaseArrayLayer(baseArrayLayer);

    vk::ImageBlit2 imageBlit;
    auto& srcSubresource =
        static_cast<vk::ImageSubresourceLayers&>(imageBlit.srcSubresource);
    auto& dstSubresource =
        static_cast<vk::ImageSubresourceLayers&>(imageBlit.dstSubresource);

    srcSubresource.setAspectMask(aspectFlags);
    srcSubresource.setLayerCount(layerCount);
    srcSubresource.setBaseArrayLayer(baseArrayLayer);
    dstSubresource.setAspectMask(aspectFlags);
    dstSubresource.setLayerCount(layerCount);
    dstSubresource.setBaseArrayLayer(baseArrayLayer);

    vk::BlitImageInfo2 blitImageInfo;
    blitImageInfo.setSrcImage(image);
    blitImageInfo.setSrcImageLayout(vk::ImageLayout::eTransferSrcOptimal);
    blitImageInfo.setDstImage(image);
    blitImageInfo.setDstImageLayout(vk::ImageLayout::eTransferDstOptimal);
    blitImageInfo.setRegionCount(1);
    blitImageInfo.setRegions(&imageBlit);
    blitImageInfo.setFilter(vk::Filter::eLinear);

    if (uint32_t i = 1; i != mipLevels) {
        vk::Offset3D mipOffset(extent.width, extent.height, extent.depth);
        if (!hostInitialized) {
            subresourceRange.baseMipLevel = i - 1;
            cmdPipelineBarriers(barrier);
        }
        for (;;) {
            srcSubresource.mipLevel = subresourceRange.baseMipLevel;
            dstSubresource.mipLevel = i;
            imageBlit.srcOffsets[1] = mipOffset;
            if (mipOffset.x > 1)
                mipOffset.x >>= 1;
            if (mipOffset.y > 1)
                mipOffset.y >>= 1;
            if (mipOffset.z > 1)
                mipOffset.z >>= 1;
            imageBlit.dstOffsets[1] = mipOffset;
            cmdBlitImage2(blitImageInfo);
            if (++i == mipLevels)
                break;
            subresourceRange.baseMipLevel = i - 1;
            cmdPipelineBarriers(barrier);
        }
    }
    barrier.setDstStageMask(vk::PipelineStageFlagBits2::bFragmentShader);

    subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
    barrier.setDstAccessMask(vk::AccessFlagBits2::bShaderRead);
    cmdPipelineBarriers(barrier);

    subresourceRange.setLevelCount(subresourceRange.baseMipLevel);
    subresourceRange.baseMipLevel = 0;
    barrier.setOldLayout(vk::ImageLayout::eTransferSrcOptimal);
    barrier.setSrcAccessMask(vk::AccessFlagBits2::bTransferRead);
    cmdPipelineBarriers(barrier);
}

bool vx::PhysicalDeviceInfo::init(PhysicalDevice physicalDevice) {
    physicalDevice.getProperties(&properties);
    queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
    if (!physicalDevice.enumerateDeviceExtensionProperties().extract(
            extensionProperties)) {
        return false;
    }
    std::ranges::sort(extensionProperties, std::ranges::less{},
                      [](const vk::ExtensionProperties& props) {
                          return props.getExtensionName();
                      });
    return true;
}

bool vx::PhysicalDeviceInfo::hasExtension(
    std::string_view name) const noexcept {
    const auto it =
        std::ranges::lower_bound(extensionProperties, name, std::ranges::less{},
                                 [](const vk::ExtensionProperties& props) {
                                     return props.getExtensionName();
                                 });
    return it != extensionProperties.end() && it->getExtensionName() == name;
}
