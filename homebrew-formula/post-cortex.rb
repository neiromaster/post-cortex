class PostCortex < Formula
  desc "Production-grade intelligent conversation memory system for AI assistants"
  homepage "https://github.com/julymetodiev/post-cortex"
  version "0.1.0"
  license "MIT"

  on_macos do
    if Hardware::CPU.intel?
      url "https://github.com/julymetodiev/post-cortex/releases/download/v#{version}/post-cortex-x86_64-apple-darwin"
      sha256 "UPDATE_AFTER_RELEASE"

      resource "daemon" do
        url "https://github.com/julymetodiev/post-cortex/releases/download/v#{version}/post-cortex-daemon-x86_64-apple-darwin"
        sha256 "UPDATE_AFTER_RELEASE"
      end
    else
      url "https://github.com/julymetodiev/post-cortex/releases/download/v#{version}/post-cortex-aarch64-apple-darwin"
      sha256 "UPDATE_AFTER_RELEASE"

      resource "daemon" do
        url "https://github.com/julymetodiev/post-cortex/releases/download/v#{version}/post-cortex-daemon-aarch64-apple-darwin"
        sha256 "UPDATE_AFTER_RELEASE"
      end
    end
  end

  on_linux do
    if Hardware::CPU.intel?
      url "https://github.com/julymetodiev/post-cortex/releases/download/v#{version}/post-cortex-x86_64-unknown-linux-gnu"
      sha256 "UPDATE_AFTER_RELEASE"

      resource "daemon" do
        url "https://github.com/julymetodiev/post-cortex/releases/download/v#{version}/post-cortex-daemon-x86_64-unknown-linux-gnu"
        sha256 "UPDATE_AFTER_RELEASE"
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
