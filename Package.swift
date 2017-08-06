import PackageDescription

let package = Package(
    name: "TFSwift",
    dependencies: [
        .Package(url: "https://github.com/PerfectlySoft/Perfect-TensorFlow.git", majorVersion: 1)
    ]
)