package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"

	"go.temporal.io/api/workflowservice/v1"
	"go.temporal.io/sdk/client"
)

const (
	TaskQueueName = "goose-task-queue"
	Namespace     = "default"
)

// PortConfig holds the port configuration for Temporal services
type PortConfig struct {
	TemporalPort int // Main Temporal server port
	UIPort       int // Temporal UI port
	HTTPPort     int // HTTP API port
}

// getManagedRecipesDir returns the proper directory for storing managed recipes
func getManagedRecipesDir() (string, error) {
	var baseDir string

	switch runtime.GOOS {
	case "darwin":
		// macOS: ~/Library/Application Support/temporal/managed-recipes
		homeDir, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("failed to get user home directory: %w", err)
		}
		baseDir = filepath.Join(homeDir, "Library", "Application Support", "temporal", "managed-recipes")
	case "linux":
		// Linux: ~/.local/share/temporal/managed-recipes
		homeDir, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("failed to get user home directory: %w", err)
		}
		baseDir = filepath.Join(homeDir, ".local", "share", "temporal", "managed-recipes")
	case "windows":
		// Windows: %APPDATA%\temporal\managed-recipes
		appDataDir := os.Getenv("APPDATA")
		if appDataDir == "" {
			homeDir, err := os.UserHomeDir()
			if err != nil {
				return "", fmt.Errorf("failed to get user home directory: %w", err)
			}
			appDataDir = filepath.Join(homeDir, "AppData", "Roaming")
		}
		baseDir = filepath.Join(appDataDir, "temporal", "managed-recipes")
	default:
		// Fallback for unknown OS
		homeDir, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("failed to get user home directory: %w", err)
		}
		baseDir = filepath.Join(homeDir, ".local", "share", "temporal", "managed-recipes")
	}

	return baseDir, nil
}

// findAvailablePort finds an available port starting from the given port
func findAvailablePort(startPort int) (int, error) {
	for port := startPort; port < startPort+100; port++ {
		ln, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
		if err == nil {
			ln.Close()
			return port, nil
		}
	}
	return 0, fmt.Errorf("no available port found starting from %d", startPort)
}

// findAvailablePorts finds available ports for all Temporal services
func findAvailablePorts() (*PortConfig, error) {
	// Try to find available ports starting from preferred defaults
	temporalPort, err := findAvailablePort(7233)
	if err != nil {
		return nil, fmt.Errorf("failed to find available port for Temporal server: %w", err)
	}

	uiPort, err := findAvailablePort(8233)
	if err != nil {
		return nil, fmt.Errorf("failed to find available port for Temporal UI: %w", err)
	}

	// For HTTP port, check environment variable first
	httpPort := 8080
	if portEnv := os.Getenv("PORT"); portEnv != "" {
		if parsed, err := strconv.Atoi(portEnv); err == nil {
			httpPort = parsed
		}
	}

	// Verify HTTP port is available, find alternative if not
	finalHTTPPort, err := findAvailablePort(httpPort)
	if err != nil {
		return nil, fmt.Errorf("failed to find available port for HTTP server: %w", err)
	}

	return &PortConfig{
		TemporalPort: temporalPort,
		UIPort:       uiPort,
		HTTPPort:     finalHTTPPort,
	}, nil
}

// isTemporalServerRunning checks if Temporal server is accessible
func isTemporalServerRunning(port int) bool {
	// Try to create a client connection to check if server is running
	c, err := client.Dial(client.Options{
		HostPort:  fmt.Sprintf("127.0.0.1:%d", port),
		Namespace: Namespace,
	})
	if err != nil {
		return false
	}
	defer c.Close()

	// Try a simple operation to verify the connection works
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	_, err = c.WorkflowService().GetSystemInfo(ctx, &workflowservice.GetSystemInfoRequest{})
	return err == nil
}

// findTemporalCLI attempts to find the temporal CLI binary
func findTemporalCLI() (string, error) {
	log.Println("Looking for temporal CLI binary...")

	// First, try to find temporal in PATH using exec.LookPath
	log.Println("Checking PATH for temporal CLI...")
	if path, err := exec.LookPath("temporal"); err == nil {
		log.Printf("Found temporal in PATH at: %s", path)
		// Verify it's the correct temporal CLI by checking version
		log.Println("Verifying temporal CLI version...")
		cmd := exec.Command(path, "--version")
		if err := cmd.Run(); err == nil {
			log.Printf("Successfully verified temporal CLI at: %s", path)
			return path, nil
		} else {
			log.Printf("Failed to verify temporal CLI at %s: %v", path, err)
		}
	} else {
		log.Printf("temporal not found in PATH: %v", err)
	}

	// Try using 'which' command to find temporal
	cmd := exec.Command("which", "temporal")
	if output, err := cmd.Output(); err == nil {
		path := strings.TrimSpace(string(output))
		if path != "" {
			// Verify it's the correct temporal CLI by checking version
			cmd := exec.Command(path, "--version")
			if err := cmd.Run(); err == nil {
				return path, nil
			}
		}
	}

	// If not found in PATH, try different possible locations for the temporal CLI
	log.Println("Checking bundled/local locations for temporal CLI...")
	possiblePaths := []string{
		"./temporal", // Current directory
	}

	// Also try relative to the current executable (most important for bundled apps)
	if exePath, err := os.Executable(); err == nil {
		exeDir := filepath.Dir(exePath)
		log.Printf("Executable directory: %s", exeDir)
		additionalPaths := []string{
			filepath.Join(exeDir, "temporal"),
			filepath.Join(exeDir, "temporal.exe"), // Windows
			// Also try one level up (for development)
			filepath.Join(exeDir, "..", "temporal"),
			filepath.Join(exeDir, "..", "temporal.exe"),
		}
		possiblePaths = append(possiblePaths, additionalPaths...)
		log.Printf("Will check these additional paths: %v", additionalPaths)
	} else {
		log.Printf("Failed to get executable path: %v", err)
	}

	log.Printf("Checking %d possible paths for temporal CLI", len(possiblePaths))

	// Test each possible path
	for i, path := range possiblePaths {
		log.Printf("Checking path %d/%d: %s", i+1, len(possiblePaths), path)
		if _, err := os.Stat(path); err == nil {
			log.Printf("File exists at: %s", path)
			// File exists, test if it's executable and the right binary
			cmd := exec.Command(path, "--version")
			if err := cmd.Run(); err == nil {
				log.Printf("Successfully verified temporal CLI at: %s", path)
				return path, nil
			} else {
				log.Printf("Failed to verify temporal CLI at %s: %v", path, err)
			}
		} else {
			log.Printf("File does not exist at %s: %v", path, err)
		}
	}

	return "", fmt.Errorf("temporal CLI not found in PATH or any of the expected locations: %v", possiblePaths)
}

// ensureTemporalServerRunning checks if Temporal server is running and starts it if needed
func ensureTemporalServerRunning(ports *PortConfig) error {
	log.Println("Checking if Temporal server is running...")

	// Check if Temporal server is already running by trying to connect
	if isTemporalServerRunning(ports.TemporalPort) {
		log.Printf("Temporal server is already running on port %d", ports.TemporalPort)
		return nil
	}

	log.Printf("Temporal server not running, attempting to start it on port %d...", ports.TemporalPort)

	// Find the temporal CLI binary
	temporalCmd, err := findTemporalCLI()
	if err != nil {
		log.Printf("ERROR: Could not find temporal CLI: %v", err)
		return fmt.Errorf("could not find temporal CLI: %w", err)
	}

	log.Printf("Using Temporal CLI at: %s", temporalCmd)

	// Start Temporal server in background
	args := []string{"server", "start-dev",
		"--db-filename", "temporal.db",
		"--port", strconv.Itoa(ports.TemporalPort),
		"--ui-port", strconv.Itoa(ports.UIPort),
		"--log-level", "warn"}

	log.Printf("Starting Temporal server with command: %s %v", temporalCmd, args)

	cmd := exec.Command(temporalCmd, args...)

	// Properly detach the process so it survives when the parent exits
	configureSysProcAttr(cmd)

	// Redirect stdin/stdout/stderr to avoid hanging
	cmd.Stdin = nil
	cmd.Stdout = nil
	cmd.Stderr = nil

	// Start the process
	if err := cmd.Start(); err != nil {
		log.Printf("ERROR: Failed to start Temporal server: %v", err)
		return fmt.Errorf("failed to start Temporal server: %w", err)
	}

	log.Printf("Temporal server started with PID: %d (port: %d, UI port: %d)",
		cmd.Process.Pid, ports.TemporalPort, ports.UIPort)

	// Wait for server to be ready (with timeout)
	log.Println("Waiting for Temporal server to be ready...")
	timeout := time.After(30 * time.Second)
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	attemptCount := 0
	for {
		select {
		case <-timeout:
			log.Printf("ERROR: Timeout waiting for Temporal server to start after %d attempts", attemptCount)
			return fmt.Errorf("timeout waiting for Temporal server to start")
		case <-ticker.C:
			attemptCount++
			log.Printf("Checking if Temporal server is ready (attempt %d)...", attemptCount)
			if isTemporalServerRunning(ports.TemporalPort) {
				log.Printf("Temporal server is now ready on port %d", ports.TemporalPort)
				return nil
			} else {
				log.Printf("Temporal server not ready yet (attempt %d)", attemptCount)
			}
		}
	}
}

func main() {
	log.Println("Starting Temporal service...")

	// Create Temporal service (this will find available ports automatically)
	service, err := NewTemporalService()
	if err != nil {
		log.Fatalf("Failed to create Temporal service: %v", err)
	}

	// Use the dynamically assigned HTTP port
	httpPort := service.GetHTTPPort()
	temporalPort := service.GetTemporalPort()
	uiPort := service.GetUIPort()

	log.Printf("Temporal server running on port %d", temporalPort)
	log.Printf("Temporal UI available at http://localhost:%d", uiPort)

	// Set up HTTP server
	mux := http.NewServeMux()
	mux.HandleFunc("/jobs", service.handleJobs)
	mux.HandleFunc("/health", service.handleHealth)
	mux.HandleFunc("/ports", service.handlePorts)

	server := &http.Server{
		Addr:    fmt.Sprintf(":%d", httpPort),
		Handler: mux,
	}

	// Handle graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigChan
		log.Println("Received shutdown signal")

		// Kill all managed processes first
		globalProcessManager.KillAllProcesses()

		// Shutdown HTTP server
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		server.Shutdown(ctx)

		// Stop Temporal service
		service.Stop()

		os.Exit(0)
	}()

	log.Printf("Temporal service starting on port %d", httpPort)
	log.Printf("Health endpoint: http://localhost:%d/health", httpPort)
	log.Printf("Jobs endpoint: http://localhost:%d/jobs", httpPort)
	log.Printf("Ports endpoint: http://localhost:%d/ports", httpPort)

	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("HTTP server failed: %v", err)
	}
}
