import AppKit
import Foundation

final class SetupApp: NSObject, NSApplicationDelegate, NSWindowDelegate {
    private var window: NSWindow!
    private let log = NSTextView()
    private let state = NSTextField(labelWithString: "Ready")
    private let sourceLabel = NSTextField(labelWithString: "")
    private let modelsLabel = NSTextField(labelWithString: "")
    private var controls: [NSButton] = []
    private var stopButton: NSButton!
    private var process: Process?
    private var output: Pipe?
    private var isServer = false
    private var quitting = false
    private var root = Bundle.main.bundleURL.deletingLastPathComponent().deletingLastPathComponent()
    private var models = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("strata-models")

    func applicationDidFinishLaunching(_ notification: Notification) {
        if let path = UserDefaults.standard.string(forKey: "SourceFolder") { root = URL(fileURLWithPath: path) }
        if let path = UserDefaults.standard.string(forKey: "ModelFolder") { models = URL(fileURLWithPath: path) }
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 800, height: 590),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
        window.title = "Strata Setup - M1 Max / M2 Max"
        window.delegate = self
        window.isReleasedWhenClosed = false
        window.minSize = NSSize(width: 720, height: 500)
        window.center()
        let stack = NSStackView()
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 12
        stack.translatesAutoresizingMaskIntoConstraints = false
        window.contentView!.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: window.contentView!.leadingAnchor, constant: 20),
            stack.trailingAnchor.constraint(equalTo: window.contentView!.trailingAnchor, constant: -20),
            stack.topAnchor.constraint(equalTo: window.contentView!.topAnchor, constant: 20),
            stack.bottomAnchor.constraint(equalTo: window.contentView!.bottomAnchor, constant: -20)
        ])
        let intro = NSTextField(wrappingLabelWithString:
            "Source-built, local-only Qwen server. Requires M1 Max / M2 Max, 64 GB RAM and macOS 15+. No automatic updates.")
        stack.addArrangedSubview(intro)
        sourceLabel.lineBreakMode = .byTruncatingMiddle
        modelsLabel.lineBreakMode = .byTruncatingMiddle
        stack.addArrangedSubview(sourceLabel)
        stack.addArrangedSubview(modelsLabel)
        let folders = NSStackView(views: [button("Source Folder...", #selector(selectSource)),
                                         button("Model Folder...", #selector(selectModels))])
        stack.addArrangedSubview(folders)
        let actions = NSStackView(views: [button("Check Setup", #selector(check)),
            button("Install from Source", #selector(install)), button("GPU Memory...", #selector(gpuMemory)),
            button("Start", #selector(start)), button("Status", #selector(status))])
        stopButton = NSButton(title: "Stop", target: self, action: #selector(stop))
        stopButton.isEnabled = false
        actions.addArrangedSubview(stopButton)
        stack.addArrangedSubview(actions)
        stack.addArrangedSubview(state)
        let scroll = NSScrollView()
        scroll.hasVerticalScroller = true
        scroll.borderType = .bezelBorder
        log.isEditable = false
        log.isSelectable = true
        log.font = NSFont.systemFont(ofSize: 12)
        log.autoresizingMask = [.width]
        log.textContainer?.widthTracksTextView = true
        scroll.documentView = log
        stack.addArrangedSubview(scroll)
        scroll.widthAnchor.constraint(equalTo: stack.widthAnchor).isActive = true
        scroll.heightAnchor.constraint(greaterThanOrEqualToConstant: 230).isActive = true
        for view in [intro, sourceLabel, modelsLabel] {
            view.widthAnchor.constraint(equalTo: stack.widthAnchor).isActive = true
        }
        refreshLabels()
        append("Install runs the repository's existing verified setup scripts. It builds the engine and Python packages from source.\n")
        append("Keep the source folder in place: this app and the engine use it at runtime. Stop other GPU model servers before Start.\n")
        if !validRoot() { append("Choose the source checkout with Source Folder before continuing.\n") }
        let menu = NSMenu()
        let appItem = NSMenuItem()
        menu.addItem(appItem)
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "Quit Strata Setup", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appItem.submenu = appMenu
        NSApp.mainMenu = menu
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    private func button(_ title: String, _ action: Selector) -> NSButton {
        let b = NSButton(title: title, target: self, action: action)
        controls.append(b)
        return b
    }

    private func validRoot() -> Bool {
        FileManager.default.fileExists(atPath: root.appendingPathComponent("scripts/setup-app.sh").path)
    }

    private func refreshLabels() {
        sourceLabel.stringValue = "Source: " + root.path
        modelsLabel.stringValue = "Models for installation: " + models.path
    }

    private func append(_ text: String) {
        log.textStorage?.append(NSAttributedString(string: text))
        if log.string.utf16.count > 250_000 {
            log.textStorage?.deleteCharacters(in: NSRange(location: 0, length: 50_000))
        }
        log.scrollToEndOfDocument(nil)
    }

    private func confirm(_ title: String, _ text: String) -> Bool {
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = text
        alert.addButton(withTitle: "Continue")
        alert.addButton(withTitle: "Cancel")
        return alert.runModal() == .alertFirstButtonReturn
    }

    @objc private func selectSource() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            let previous = root
            root = url
            if !validRoot() {
                root = previous
                append("That folder is not a Strata source checkout with scripts/setup-app.sh.\n")
                return
            }
            UserDefaults.standard.set(root.path, forKey: "SourceFolder")
            refreshLabels()
        }
    }

    @objc private func selectModels() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            models = url
            UserDefaults.standard.set(models.path, forKey: "ModelFolder")
            refreshLabels()
            append("Existing strata-cached.json paths are preserved. Selecting a folder only changes where Install prepares models.\n")
        }
    }

    private func run(_ arguments: [String], server: Bool = false) {
        guard process == nil else { return }
        guard validRoot() else { append("Choose the source checkout first.\n"); return }
        let p = Process()
        let pipe = Pipe()
        p.executableURL = URL(fileURLWithPath: "/bin/bash")
        p.arguments = arguments
        p.currentDirectoryURL = root
        var env = ProcessInfo.processInfo.environment
        env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
        env["STRATA_MODELS"] = models.path
        env["STRATA_PORT"] = "8000"
        env["PYTHONUNBUFFERED"] = "1"
        p.environment = env
        p.standardInput = FileHandle.nullDevice
        p.standardOutput = pipe
        p.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            let text = String(decoding: data, as: UTF8.self)
            DispatchQueue.main.async { self?.append(text) }
        }
        p.terminationHandler = { [weak self] child in
            pipe.fileHandleForReading.readabilityHandler = nil
            let tail = pipe.fileHandleForReading.readDataToEndOfFile()
            DispatchQueue.main.async {
                guard let self = self else { return }
                if !tail.isEmpty { self.append(String(decoding: tail, as: UTF8.self)) }
                self.append("\nFinished (exit \(child.terminationStatus)).\n")
                self.process = nil
                self.output = nil
                self.isServer = false
                self.controls.forEach { $0.isEnabled = true }
                self.stopButton.isEnabled = false
                self.state.stringValue = child.terminationStatus == 0 ? "Ready" : "Stopped - see log"
                if self.quitting { NSApp.reply(toApplicationShouldTerminate: true) }
            }
        }
        do {
            try p.run()
            process = p
            output = pipe
            isServer = server
            controls.forEach { $0.isEnabled = false }
            // Status remains available while this app owns the server.
            if server { controls.last?.isEnabled = true }
            stopButton.isEnabled = server
            state.stringValue = server ? "Server starting - use Status to check readiness" : "Working - see log"
        } catch {
            pipe.fileHandleForReading.readabilityHandler = nil
            append("Could not start: \(error.localizedDescription)\n")
        }
    }

    private func stage(_ name: String, server: Bool = false) {
        run([root.appendingPathComponent("scripts/setup-app.sh").path, name], server: server)
    }

    @objc private func check() { stage("check") }

    @objc private func install() {
        guard confirm("Build and install from source?",
            "This downloads pinned build dependencies, Python source packages and about 76 GB of model shards, plus draft weights. Packs and caches require more disk space. Downloads use the existing scripts and can reuse completed files. No administrator changes are made by Install. Existing config is kept. Review Check Setup first. Setup cannot be cancelled safely from this app once started.") else { return }
        stage("install")
    }

    @objc private func gpuMemory() {
        guard confirm("Allow the GPU to use 58000 MiB?",
            "This changes iogpu.wired_limit_mb until the next reboot. It leaves less memory for other apps. Close other GPU model servers first. macOS will ask for administrator permission. No login service is installed.") else { return }
        guard validRoot() else { append("Choose the source checkout first.\n"); return }
        // The privileged command is fixed and contains no paths or user input.
        run([root.appendingPathComponent("scripts/setup-app-memory.sh").path])
    }

    @objc private func start() { stage("start", server: true) }

    @objc private func stop() {
        guard isServer, let p = process, p.isRunning else { return }
        append("Stopping the server started by this app; waiting for engine cleanup.\n")
        stopButton.isEnabled = false
        p.terminate()
    }

    @objc private func status() {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:8000/v1/models")!)
        request.timeoutInterval = 5
        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            var message = "No server response: \(error?.localizedDescription ?? "unknown error")\n"
            if let http = response as? HTTPURLResponse, let data = data {
                message = "Status HTTP \(http.statusCode): \(String(decoding: data, as: UTF8.self))\n"
            }
            DispatchQueue.main.async { self?.append(message) }
        }.resume()
    }

    func windowShouldClose(_ sender: NSWindow) -> Bool {
        if let p = process, p.isRunning, !isServer {
            append("Wait for setup to finish before closing this window.\n")
            return false
        }
        return true
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let p = process, p.isRunning else { return .terminateNow }
        guard isServer else {
            append("Wait for setup to finish before quitting. Closing the app must not leave an unfinished build running.\n")
            window.makeKeyAndOrderFront(nil)
            return .terminateCancel
        }
        quitting = true
        stop()
        return .terminateLater
    }
}

let app = NSApplication.shared
let delegate = SetupApp()
app.setActivationPolicy(.regular)
app.delegate = delegate
app.run()
