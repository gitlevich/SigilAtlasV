import Foundation

// sa-photos: macOS-only helper that talks to PhotoKit on behalf of the
// Tauri app. Stdout is NDJSON (one JSON object per line). Stderr is human
// log output. Exit codes are stable: see ExitCode.

let usage = """
sa-photos — PhotoKit bridge for Sigil Atlas

usage:
  sa-photos auth                          Request / report library authorization.
  sa-photos enumerate [--since <epoch>]   Stream one NDJSON record per photo to stdout.
  sa-photos thumb <id> <size> <outPath>   Write a single JPEG thumbnail to outPath.
  sa-photos thumbs                        Read NDJSON work items from stdin; write JPEGs in parallel.

exit codes:
  0 ok              2 denied          3 restricted      4 not-determined
  5 not-found       6 io-error        64 usage
"""

let args = Array(CommandLine.arguments.dropFirst())
guard let command = args.first else {
    FileHandle.standardError.write(Data(usage.utf8))
    exit(ExitCode.usage)
}

let rest = Array(args.dropFirst())

switch command {
case "auth":
    AuthCommand().run()
case "enumerate":
    EnumerateCommand().run(args: rest)
case "thumb":
    ThumbCommand().run(args: rest)
case "thumbs":
    ThumbsCommand().run()
case "-h", "--help", "help":
    print(usage)
    exit(0)
default:
    FileHandle.standardError.write(Data("unknown command: \(command)\n\n\(usage)".utf8))
    exit(ExitCode.usage)
}
