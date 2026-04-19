import Foundation
import Photos

// Request (or check) read authorization for the user's Photos library. Prints
// a single lowercase status token to stdout: authorized, limited, denied,
// restricted, notdetermined. Exits 0 on authorized or limited, non-zero
// otherwise so callers can pipeline `auth && enumerate ...`.
struct AuthCommand {
    func run() -> Never {
        let current = PHPhotoLibrary.authorizationStatus(for: .readWrite)
        if current == .notDetermined {
            let sem = DispatchSemaphore(value: 0)
            var granted: PHAuthorizationStatus = .notDetermined
            PHPhotoLibrary.requestAuthorization(for: .readWrite) { status in
                granted = status
                sem.signal()
            }
            sem.wait()
            exitFor(granted)
        } else {
            exitFor(current)
        }
    }

    private func exitFor(_ status: PHAuthorizationStatus) -> Never {
        switch status {
        case .authorized:
            print("authorized")
            exit(ExitCode.ok)
        case .limited:
            print("limited")
            exit(ExitCode.ok)
        case .denied:
            print("denied")
            exit(ExitCode.denied)
        case .restricted:
            print("restricted")
            exit(ExitCode.restricted)
        case .notDetermined:
            print("notdetermined")
            exit(ExitCode.notDetermined)
        @unknown default:
            print("unknown")
            exit(ExitCode.denied)
        }
    }
}
