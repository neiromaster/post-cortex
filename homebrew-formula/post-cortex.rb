class PostCortex < Formula
  desc "Production-grade intelligent conversation memory system for AI assistants"
  homepage "https://github.com/julymetodiev/post-cortex"
  version "0.1.0"
  license "MIT"

  on_macos do
    if Hardware::CPU.intel?
      url "https://github.com/julymetodiev/post-cortex/releases/download/v#{version}/post-cortex-x86_64-apple-darwin"
      sha256 "b4652c9dc7f5501857ab1a8cd4b75bf4b76b0e95682dd3c3174116cdc4622ddd"

      resource "daemon" do
        url "https://github.com/julymetodiev/post-cortex/releases/download/v#{version}/post-cortex-daemon-x86_64-apple-darwin"
        sha256 "da726a3e5b0fc6f7933f5248241e980f66387c312e29dda83b72052786f46593"
      end
    else
      url "https://github.com/julymetodiev/post-cortex/releases/download/v#{version}/post-cortex-aarch64-apple-darwin"
      sha256 "93062e5325c0fdc6232316690fe41d80db9543b3bf9cf48e65d2f95eb531396f"

      resource "daemon" do
        url "https://github.com/julymetodiev/post-cortex/releases/download/v#{version}/post-cortex-daemon-aarch64-apple-darwin"
        sha256 "0acc00fb276070dae83322a4330761d35bf0bac62531d7138aea088655f80e29"
      end
    end
  end

  on_linux do
    if Hardware::CPU.intel?
      url "https://github.com/julymetodiev/post-cortex/releases/download/v#{version}/post-cortex-x86_64-unknown-linux-gnu"
      sha256 "b3ed3e0a56fd836dd96d66c10b1f37026d1aae66f6b77bd8b3dbb6649e235101"

      resource "daemon" do
        url "https://github.com/julymetodiev/post-cortex/releases/download/v#{version}/post-cortex-daemon-x86_64-unknown-linux-gnu"
        sha256 "db0fa639771640759294d2c57426a81443aa41ac626f34c4ebd52454f1265577"
      end
    end
  end

  def install
    # Install post-cortex (stdio MCP server)
    if OS.mac? && Hardware::CPU.intel?
      bin.install "post-cortex-x86_64-apple-darwin" => "post-cortex"
    elsif OS.mac? && Hardware::CPU.arm?
      bin.install "post-cortex-aarch64-apple-darwin" => "post-cortex"
    elsif OS.linux?
      bin.install "post-cortex-x86_64-unknown-linux-gnu" => "post-cortex"
    end

    # Install post-cortex-daemon (HTTP daemon)
    resource("daemon").stage do
      if OS.mac? && Hardware::CPU.intel?
        bin.install "post-cortex-daemon-x86_64-apple-darwin" => "post-cortex-daemon"
      elsif OS.mac? && Hardware::CPU.arm?
        bin.install "post-cortex-daemon-aarch64-apple-darwin" => "post-cortex-daemon"
      elsif OS.linux?
        bin.install "post-cortex-daemon-x86_64-unknown-linux-gnu" => "post-cortex-daemon"
      end
    end
  end

  def caveats
    <<~EOS
      Post-Cortex has been installed with TWO binaries:

      1. post-cortex        - Stdio MCP server (simple)
      2. post-cortex-daemon - HTTP daemon (advanced)

      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      📦 STDIO MODE (Simple - Claude Desktop Integration)
      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

      Add to Claude Desktop config (~/.claude.json):

      {
        "mcpServers": {
          "post-cortex": {
            "command": "#{bin}/post-cortex"
          }
        }
      }

      Then restart Claude Desktop.

      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      🚀 DAEMON MODE (Advanced - HTTP API + Background Service)
      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

      Initialize and start daemon:
        post-cortex-daemon init
        post-cortex-daemon start

      Claude Desktop config:
      {
        "mcpServers": {
          "post-cortex": {
            "type": "sse",
            "url": "http://localhost:3737/sse"
          }
        }
      }

      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      🔧 Service Management (Optional)
      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

      For auto-start daemon on boot, see service installation:
      https://github.com/julymetodiev/post-cortex#service-management-daemon-mode

      macOS (launchd):  install/launchd/README.md
      Linux (systemd):  install/systemd/README.md

      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      📚 Full Documentation: https://github.com/julymetodiev/post-cortex
      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    EOS
  end

  test do
    # Test stdio server
    assert_predicate bin/"post-cortex", :exist?

    # Test daemon binary
    assert_predicate bin/"post-cortex-daemon", :exist?

    # Test daemon help output
    output = shell_output("#{bin}/post-cortex-daemon help 2>&1")
    assert_match "Post-Cortex Daemon", output
  end
end
