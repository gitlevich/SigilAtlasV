import Foundation
import Photos

// Streams one NDJSON record per still image in the System Photos library to
// stdout. Video, audio, and hidden assets are excluded. Live Photos are
// treated as stills (we only ever use the still component). --since filters
// by creationDate epoch seconds.
struct EnumerateCommand {
    func run(args: [String]) -> Never {
        let since = parseSince(args)

        let status = PHPhotoLibrary.authorizationStatus(for: .readWrite)
        guard status == .authorized || status == .limited else {
            emitError("sa-photos: not authorized (status=\(status.rawValue)); run `sa-photos auth` first")
            exit(ExitCode.denied)
        }

        let options = PHFetchOptions()
        var predicates: [NSPredicate] = [
            NSPredicate(format: "mediaType = %d", PHAssetMediaType.image.rawValue)
        ]
        if let since {
            predicates.append(NSPredicate(format: "creationDate > %@", since as NSDate))
        }
        options.predicate = NSCompoundPredicate(andPredicateWithSubpredicates: predicates)
        options.sortDescriptors = [
            NSSortDescriptor(key: "creationDate", ascending: true)
        ]

        let result = PHAsset.fetchAssets(with: options)
        // Count-first line lets the caller size progress bars before records
        // start flowing. Total == result.count because the predicate already
        // filtered to still images.
        emitNDJSON(["type": "count", "total": result.count])

        var count = 0
        result.enumerateObjects { asset, _, _ in
            emitNDJSON(record(for: asset))
            count += 1
        }
        emitError("sa-photos: emitted \(count) records")
        exit(ExitCode.ok)
    }

    private func parseSince(_ args: [String]) -> Date? {
        guard let idx = args.firstIndex(of: "--since"), idx + 1 < args.count,
              let epoch = Double(args[idx + 1]) else { return nil }
        return Date(timeIntervalSince1970: epoch)
    }

    private func record(for asset: PHAsset) -> [String: Any] {
        var obj: [String: Any] = [
            "id": asset.localIdentifier,
            "w": asset.pixelWidth,
            "h": asset.pixelHeight,
            "is_live": asset.mediaSubtypes.contains(.photoLive),
            "is_screenshot": asset.mediaSubtypes.contains(.photoScreenshot),
            "favorite": asset.isFavorite
        ]
        if let date = asset.creationDate {
            obj["capture_date"] = date.timeIntervalSince1970
        }
        if let loc = asset.location {
            obj["lat"] = loc.coordinate.latitude
            obj["lon"] = loc.coordinate.longitude
        }
        return obj
    }
}
