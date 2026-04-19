import AppKit
import Foundation
import Photos

// Batched, parallel thumbnail generation. Reads one NDJSON work item per line
// on stdin — {"id": <localIdentifier>, "size": <px>, "out": <absPath>} —
// and emits one NDJSON result per line on stdout:
//   {"out": <absPath>, "ok": true}
//   {"out": <absPath>, "ok": false, "err": "<reason>"}
//
// PHAssets are fetched in one bulk call, then run through PHImageManager with
// synchronous requests dispatched in parallel via GCD concurrentPerform.
// Exits 0 unless authorization fails; individual per-item errors are reported
// in-band so the caller can reconcile which thumbnails landed.
struct ThumbsCommand {
    struct WorkItem {
        let id: String
        let size: Int
        let out: String
    }

    func run() -> Never {
        let status = PHPhotoLibrary.authorizationStatus(for: .readWrite)
        guard status == .authorized || status == .limited else {
            emitError("sa-photos: not authorized (status=\(status.rawValue))")
            exit(ExitCode.denied)
        }

        let items = readWorkItems()
        if items.isEmpty {
            exit(ExitCode.ok)
        }

        let byId = fetchAssetsById(items.map { $0.id })
        let outputLock = NSLock()

        DispatchQueue.concurrentPerform(iterations: items.count) { i in
            let item = items[i]
            let result = process(item: item, asset: byId[item.id])
            outputLock.lock()
            emitNDJSON(result)
            outputLock.unlock()
        }
        exit(ExitCode.ok)
    }

    private func readWorkItems() -> [WorkItem] {
        var items: [WorkItem] = []
        while let line = readLine(strippingNewline: true) {
            guard let data = line.data(using: .utf8),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let id = obj["id"] as? String,
                  let size = obj["size"] as? Int,
                  let out = obj["out"] as? String else {
                continue
            }
            items.append(WorkItem(id: id, size: size, out: out))
        }
        return items
    }

    private func fetchAssetsById(_ ids: [String]) -> [String: PHAsset] {
        let fetched = PHAsset.fetchAssets(withLocalIdentifiers: ids, options: nil)
        var byId: [String: PHAsset] = [:]
        fetched.enumerateObjects { asset, _, _ in
            byId[asset.localIdentifier] = asset
        }
        return byId
    }

    private func process(item: WorkItem, asset: PHAsset?) -> [String: Any] {
        guard let asset else {
            return ["out": item.out, "ok": false, "err": "not-found"]
        }

        let options = PHImageRequestOptions()
        options.deliveryMode = .highQualityFormat
        options.resizeMode = .exact
        options.isNetworkAccessAllowed = false
        options.isSynchronous = true

        let target = CGSize(width: item.size, height: item.size)
        var image: NSImage?
        var info: [AnyHashable: Any]?
        PHImageManager.default().requestImage(
            for: asset,
            targetSize: target,
            contentMode: .aspectFit,
            options: options
        ) { img, inf in
            image = img
            info = inf
        }

        if (info?[PHImageErrorKey] as? NSError) != nil || image == nil {
            return ["out": item.out, "ok": false, "err": "unavailable"]
        }

        guard let image,
              let tiff = image.tiffRepresentation,
              let rep = NSBitmapImageRep(data: tiff),
              let jpeg = rep.representation(using: .jpeg, properties: [.compressionFactor: 0.85]) else {
            return ["out": item.out, "ok": false, "err": "encode-failed"]
        }

        do {
            try jpeg.write(to: URL(fileURLWithPath: item.out))
            return ["out": item.out, "ok": true]
        } catch {
            return ["out": item.out, "ok": false, "err": "write-failed: \(error.localizedDescription)"]
        }
    }
}
