import Foundation

// One line of NDJSON to stdout, flushed immediately so a streaming consumer
// (the Tauri proxy) sees records as they are produced on large libraries.
func emitNDJSON(_ object: [String: Any]) {
    guard let data = try? JSONSerialization.data(
        withJSONObject: object,
        options: [.sortedKeys, .withoutEscapingSlashes]
    ) else { return }
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data([0x0A]))  // newline
}

func emitError(_ message: String) {
    FileHandle.standardError.write(Data("\(message)\n".utf8))
}
