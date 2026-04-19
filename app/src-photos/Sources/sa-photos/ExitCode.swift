import Foundation

enum ExitCode {
    static let ok: Int32 = 0
    static let denied: Int32 = 2
    static let restricted: Int32 = 3
    static let notDetermined: Int32 = 4
    static let notFound: Int32 = 5
    static let ioError: Int32 = 6
    static let usage: Int32 = 64
}
