import AppKit
import Foundation
import Photos

// Writes a JPEG thumbnail for a single PHAsset to outPath. Network access is
// disabled, so iCloud-only assets return the degraded cached rendition if
// available or fail with exit 5. The file format is JPEG (not PNG) to match
// FolderSource's thumbnail output and downstream readers.
struct ThumbCommand {
    func run(args: [String]) -> Never {
        guard args.count == 3,
              let size = Int(args[1]) else {
            emitError("usage: sa-photos thumb <id> <size> <outPath>")
            exit(ExitCode.usage)
        }
        let id = args[0]
        let outPath = args[2]

        let assets = PHAsset.fetchAssets(withLocalIdentifiers: [id], options: nil)
        guard let asset = assets.firstObject else {
            emitError("sa-photos: asset not found: \(id)")
            exit(ExitCode.notFound)
        }

        let options = PHImageRequestOptions()
        options.deliveryMode = .highQualityFormat
        options.resizeMode = .exact
        options.isNetworkAccessAllowed = false
        options.isSynchronous = true

        let target = CGSize(width: size, height: size)
        var resultImage: NSImage?
        var resultInfo: [AnyHashable: Any]?
        PHImageManager.default().requestImage(
            for: asset,
            targetSize: target,
            contentMode: .aspectFit,
            options: options
        ) { image, info in
            resultImage = image
            resultInfo = info
        }

        if (resultInfo?[PHImageErrorKey] as? NSError) != nil || resultImage == nil {
            emitError("sa-photos: thumbnail unavailable (likely iCloud-only): \(id)")
            exit(ExitCode.notFound)
        }

        guard let image = resultImage,
              let tiff = image.tiffRepresentation,
              let rep = NSBitmapImageRep(data: tiff),
              let jpeg = rep.representation(using: .jpeg, properties: [.compressionFactor: 0.85]) else {
            emitError("sa-photos: encode failed: \(id)")
            exit(ExitCode.ioError)
        }

        do {
            try jpeg.write(to: URL(fileURLWithPath: outPath))
        } catch {
            emitError("sa-photos: write failed: \(outPath): \(error)")
            exit(ExitCode.ioError)
        }
        exit(ExitCode.ok)
    }
}
