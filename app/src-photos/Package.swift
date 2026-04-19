// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "sa-photos",
    platforms: [.macOS(.v12)],
    targets: [
        .executableTarget(
            name: "sa-photos",
            path: "Sources/sa-photos"
        )
    ]
)
