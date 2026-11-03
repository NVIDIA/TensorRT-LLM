// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// This experimental local CLI is compiled inside the pinned Snapshot agent module.
// It invokes the upstream executor without starting the Kubernetes controller.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	containerd "github.com/containerd/containerd/v2/client"
	"github.com/containerd/containerd/v2/pkg/namespaces"
	"github.com/go-logr/logr"
	specs "github.com/opencontainers/runtime-spec/specs-go"

	"github.com/ai-dynamo/snapshot/agent/internal/executor"
	"github.com/ai-dynamo/snapshot/agent/internal/logging"
	"github.com/ai-dynamo/snapshot/agent/internal/nsmount"
	"github.com/ai-dynamo/snapshot/agent/internal/pagebroker"
	snapshotruntime "github.com/ai-dynamo/snapshot/agent/internal/runtime"
	"github.com/ai-dynamo/snapshot/agent/internal/types"
	"github.com/ai-dynamo/snapshot/api/compat"
	"github.com/ai-dynamo/snapshot/api/podcontract"
)

const snapshotRevision = "4b1300a82c6676df020c201fa884cc602251e12e"
const attemptLabel = "com.nvidia.trtllm.snapshot.attempt"

var hexID = regexp.MustCompile(`^[0-9a-f]{64}$`)
var attemptID = regexp.MustCompile(`^[0-9a-f]{32}$`)

type observations struct {
	mu     sync.Mutex
	events []map[string]any
}

type evidenceSink struct {
	logr.LogSink
	observed *observations
}

func (s *evidenceSink) Info(level int, message string, keysAndValues ...any) {
	s.LogSink.Info(level, message, keysAndValues...)
	if message != "Checkpoint timing summary" && message != "Restore timing summary" && message != "PageBroker GPU participant complete" {
		return
	}
	event := map[string]any{"message": message}
	for i := 0; i+1 < len(keysAndValues); i += 2 {
		if key, ok := keysAndValues[i].(string); ok {
			event[key] = keysAndValues[i+1]
		}
	}
	s.observed.mu.Lock()
	s.observed.events = append(s.observed.events, event)
	s.observed.mu.Unlock()
}

func (s *evidenceSink) WithName(name string) logr.LogSink {
	return &evidenceSink{LogSink: s.LogSink.WithName(name), observed: s.observed}
}

func (s *evidenceSink) WithValues(values ...any) logr.LogSink {
	return &evidenceSink{LogSink: s.LogSink.WithValues(values...), observed: s.observed}
}

type request struct {
	Operation      string `json:"operation"`
	Attempt        string `json:"attempt"`
	Artifact       string `json:"artifact"`
	Container      string `json:"container"`
	Image          string `json:"image"`
	Containerd     string `json:"containerd_socket"`
	TimeoutSeconds int    `json:"timeout_seconds"`
	Interpose      bool   `json:"interpose"`
}

func (r request) validate() error {
	if r.Operation != "capture" && r.Operation != "restore" && r.Operation != "inspect" {
		return errors.New("operation must be inspect, capture or restore")
	}
	if !attemptID.MatchString(r.Attempt) || !attemptID.MatchString(r.Artifact) || !hexID.MatchString(r.Container) {
		return errors.New("require fresh attempt/artifact IDs and a full immutable container ID")
	}
	if len(r.Image) != 71 || r.Image[:7] != "sha256:" || !hexID.MatchString(r.Image[7:]) {
		return errors.New("require immutable Docker image ID")
	}
	if r.Containerd != "/run/docker/containerd/containerd.sock" && r.Containerd != "/run/containerd/containerd.sock" {
		return errors.New("containerd socket is outside the experimental allowlist")
	}
	if r.TimeoutSeconds < 1 || r.TimeoutSeconds > 7200 {
		return errors.New("timeout must be between 1 and 7200 seconds")
	}
	return nil
}

type containerInfo struct {
	ID    string `json:"Id"`
	Image string
	State struct {
		Running bool
		Pid     int
	}
	Config struct {
		Labels map[string]string
	}
	HostConfig struct {
		NetworkMode   string
		PidMode       string
		Privileged    bool
		NanoCpus      int64
		Memory        int64
		CpuPeriod     int64
		CpuQuota      int64
		RestartPolicy struct{ Name string }
	}
}

// dockerRuntime restricts every operation to one caller-owned immutable target.
// Docker provides ownership/image identity; containerd provides the actual OCI spec.
type dockerRuntime struct {
	request request
	client  *containerd.Client
	http    *http.Client
}

func newRuntime(r request) (*dockerRuntime, error) {
	c, err := containerd.New(r.Containerd)
	if err != nil {
		return nil, err
	}
	transport := &http.Transport{DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
		return (&net.Dialer{}).DialContext(ctx, "unix", "/var/run/docker.sock")
	}}
	return &dockerRuntime{r, c, &http.Client{Transport: transport, Timeout: 15 * time.Second}}, nil
}

func (r *dockerRuntime) call(ctx context.Context, method, path string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, method, "http://docker"+path, nil)
	if err != nil {
		return nil, err
	}
	return r.http.Do(req)
}

func (r *dockerRuntime) inspect(ctx context.Context, id string) (containerInfo, error) {
	var info containerInfo
	if id != r.request.Container {
		return info, errors.New("target is not owned by this operation")
	}
	resp, err := r.call(ctx, "GET", "/containers/"+id+"/json")
	if err != nil {
		return info, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return info, fmt.Errorf("container inspection: HTTP %d", resp.StatusCode)
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 4<<20)).Decode(&info); err != nil {
		return info, err
	}
	if info.ID != id || info.Image != r.request.Image || info.Config.Labels[attemptLabel] != r.request.Attempt {
		return info, errors.New("container ownership or immutable image mismatch")
	}
	if info.HostConfig.NetworkMode != "none" || info.HostConfig.PidMode != "" || info.HostConfig.Privileged || info.HostConfig.RestartPolicy.Name != "no" {
		return info, errors.New("workload requires isolated PID/network namespaces, no privilege and no restart policy")
	}
	return info, nil
}

func (r *dockerRuntime) ResolveContainer(ctx context.Context, id string) (int, *specs.Spec, error) {
	info, err := r.inspect(ctx, id)
	if err != nil {
		return 0, nil, err
	}
	if !info.State.Running || info.State.Pid <= 1 {
		return 0, nil, errors.New("target is not running")
	}
	ctx = namespaces.WithNamespace(ctx, "moby")
	c, err := r.client.LoadContainer(ctx, id)
	if err != nil {
		return 0, nil, err
	}
	task, err := c.Task(ctx, nil)
	if err != nil {
		return 0, nil, err
	}
	if int(task.Pid()) != info.State.Pid {
		return 0, nil, errors.New("Docker/containerd process identity mismatch")
	}
	spec, err := c.Spec(ctx)
	if err != nil {
		return 0, nil, err
	}
	if spec.Process == nil {
		return 0, nil, errors.New("missing OCI process")
	}
	// Explicit UUID selection avoids the upstream Kubernetes allocation-discovery path.
	found := false
	for _, value := range spec.Process.Env {
		if strings.HasPrefix(value, "NVIDIA_VISIBLE_DEVICES=GPU-") {
			found = true
		}
	}
	if !found {
		return 0, nil, errors.New("workload must explicitly select NVIDIA_VISIBLE_DEVICES GPU UUIDs")
	}
	if !slices.Contains(spec.Process.Env, "SNAPSHOT_CONTROL_DIR=/snapshot-control") {
		return 0, nil, errors.New("workload must bind SNAPSHOT_CONTROL_DIR to /snapshot-control")
	}
	controlMount := false
	for _, mount := range spec.Mounts {
		if mount.Destination == "/snapshot-control" && mount.Type == "bind" && !slices.Contains(mount.Options, "ro") {
			controlMount = true
		}
	}
	if !controlMount {
		return 0, nil, errors.New("control directory must be a dedicated writable bind mount")
	}
	if err := validateSeccomp(spec); err != nil {
		return 0, nil, err
	}
	return info.State.Pid, spec, nil
}

func validateSeccomp(spec *specs.Spec) error {
	if spec.Linux == nil || spec.Linux.Seccomp == nil {
		return errors.New("workload must block io_uring with seccomp")
	}
	policy := spec.Linux.Seccomp
	for _, syscall := range []string{"io_uring_setup", "io_uring_enter", "io_uring_register"} {
		blocked := policy.DefaultAction == specs.ActErrno || policy.DefaultAction == specs.ActKill || policy.DefaultAction == specs.ActKillProcess
		for _, rule := range policy.Syscalls {
			if slices.Contains(rule.Names, syscall) {
				if rule.Action == specs.ActAllow || len(rule.Args) != 0 {
					return fmt.Errorf("seccomp may permit %s", syscall)
				}
				blocked = rule.Action == specs.ActErrno || rule.Action == specs.ActKill || rule.Action == specs.ActKillProcess
			}
		}
		if !blocked {
			return fmt.Errorf("seccomp does not block %s", syscall)
		}
	}
	return nil
}

func validateHostNamespaces() error {
	for _, namespace := range []string{"pid", "cgroup"} {
		self, err := os.Readlink("/proc/self/ns/" + namespace)
		if err != nil {
			return err
		}
		host, err := os.Readlink("/host/proc/1/ns/" + namespace)
		if err != nil {
			return err
		}
		if self != host {
			return fmt.Errorf("host driver must join host %s namespace", namespace)
		}
	}
	for _, path := range []string{"/sys/fs/cgroup", "/sys/kernel/security", "/checkpoints", "/pagebroker/control", "/snapshot-binaries"} {
		info, err := os.Stat(path)
		if err != nil {
			return err
		}
		if !info.IsDir() {
			return fmt.Errorf("required mount/bundle is not a directory: %s", path)
		}
	}
	return nil
}

func environment(info containerInfo, cfg *types.AgentConfig) (compat.Environment, error) {
	if info.HostConfig.CpuQuota != 0 || info.HostConfig.CpuPeriod != 0 {
		return compat.Environment{}, errors.New("use Docker --cpus for an explicit comparable CPU budget")
	}
	return compat.Environment{ImageID: info.Image, CPUArch: runtime.GOARCH,
		CPUFeatures: cfg.HostCPUFeatures, KernelVersion: cfg.HostKernelVersion,
		CPULimit:    strconv.FormatFloat(float64(info.HostConfig.NanoCpus)/1e9, 'f', -1, 64),
		MemoryLimit: strconv.FormatInt(info.HostConfig.Memory, 10)}, nil
}

func claimTarget(root, target string) error {
	if err := os.MkdirAll(root, 0700); err != nil {
		return err
	}
	file, err := os.OpenFile(filepath.Join(root, target), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return fmt.Errorf("target already consumed or unavailable: %w", err)
	}
	if err := file.Sync(); err != nil {
		file.Close()
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	dir, err := os.Open(root)
	if err != nil {
		return err
	}
	defer dir.Close()
	return dir.Sync()
}

func (r *dockerRuntime) ResolveContainerImageID(ctx context.Context, id string) (string, error) {
	info, err := r.inspect(ctx, id)
	return info.Image, err
}

func (*dockerRuntime) ResolveContainerPodUID(context.Context, string) (string, error) {
	return "", errors.New("standalone containers have no Pod UID")
}

func (*dockerRuntime) ResolveContainerIDByPod(context.Context, string, string, string) (string, error) {
	return "", errors.New("Pod lookup is not supported")
}

func (*dockerRuntime) ResolveContainerByPod(context.Context, string, string, string) (int, *specs.Spec, error) {
	return 0, nil, errors.New("Pod lookup is not supported")
}

func (r *dockerRuntime) TerminateContainer(ctx context.Context, id string) error {
	info, err := r.inspect(ctx, id)
	if err != nil {
		return err
	}
	if !info.State.Running {
		return nil
	}
	resp, err := r.call(ctx, "POST", "/containers/"+id+"/kill?signal=KILL")
	if err != nil {
		return err
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusNoContent && resp.StatusCode != http.StatusConflict {
		return fmt.Errorf("container kill: HTTP %d", resp.StatusCode)
	}
	for {
		info, err = r.inspect(ctx, id)
		if err != nil {
			return err
		}
		if !info.State.Running {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(100 * time.Millisecond):
		}
	}
}

func (r *dockerRuntime) Close() error {
	r.http.CloseIdleConnections()
	return r.client.Close()
}

func configuration(timeout int) *types.AgentConfig {
	zero := 0
	return &types.AgentConfig{
		Storage:    types.StorageSpec{Type: "pvc", BasePath: "/checkpoints"},
		Overlay:    types.OverlaySettings{Exclusions: []string{"/proc", "/sys", "/dev", "*/__pycache__", "*.pyc"}},
		PageBroker: types.PageBrokerSpec{ControlSocketPath: "/pagebroker/control/pagebroker.sock", RestoreMode: "direct", StagingPath: "/pagebroker/staging"},
		Checkpoint: types.CheckpointSpec{CheckpointTimeoutSeconds: &timeout, RetryLimit: &zero},
		Restore:    types.RestoreSpec{RestoreTimeoutSeconds: timeout},
		CRIU: types.CRIUSettings{BinaryPath: "/usr/local/sbin/criu", GhostLimit: 536870912, LogLevel: 4,
			WorkDir: "/var/criu-work", ShellJob: true, TcpEstablished: true, FileLocks: true,
			OrphanPtsMaster: true, ExtUnixSk: true, LinkRemap: true, ExtMasters: true,
			ManageCgroupsMode: "soft", ImageIoMode: "direct", RstSibling: true,
			EvasiveDevices: true, ForceIrmap: true, LibDir: "/usr/local/lib/snapshot/criu-plugins", AllowUprobes: true},
	}
}

func execute(ctx context.Context, r request) (result map[string]any, retErr error) {
	if err := validateHostNamespaces(); err != nil {
		return nil, err
	}
	observed := &observations{}
	log := logr.New(&evidenceSink{LogSink: logging.ConfigureLogger("stderr").GetSink(), observed: observed})
	defer func() {
		if result != nil {
			result["upstream_events"] = observed.events
		}
	}()
	rt, err := newRuntime(r)
	if err != nil {
		return nil, err
	}
	defer rt.Close()
	pid, spec, err := rt.ResolveContainer(ctx, r.Container)
	if err != nil {
		return nil, err
	}
	result = map[string]any{"snapshot_revision": snapshotRevision, "operation": r.Operation, "attempt": r.Attempt, "artifact": r.Artifact, "container": r.Container, "host_pid": pid}
	if r.Operation == "inspect" {
		return result, nil
	}
	if r.Operation == "restore" && !slices.Equal(spec.Process.Args, []string{"/bin/sleep", "infinity"}) {
		return result, errors.New("restore destination must run only /bin/sleep infinity")
	}
	mutating := false
	defer func() {
		if retErr == nil || !mutating {
			return
		}
		cleanup, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if r.Operation == "restore" {
			retErr = errors.Join(retErr, snapshotruntime.WriteControlSentinel(pid, podcontract.RestoreFailedFile, []byte("host operation failed\n")))
		}
		retErr = errors.Join(retErr, rt.TerminateContainer(cleanup, r.Container))
	}()
	cfg := configuration(r.TimeoutSeconds)
	if err := cfg.Validate(); err != nil {
		return result, err
	}
	cfg.HostKernelVersion, err = snapshotruntime.ReadKernelVersion(snapshotruntime.HostProcPath)
	if err != nil {
		return result, err
	}
	cfg.HostCPUFeatures, err = snapshotruntime.ReadCPUFeatures(snapshotruntime.HostProcPath)
	if err != nil {
		return result, err
	}
	info, err := rt.inspect(ctx, r.Container)
	if err != nil {
		return result, err
	}
	env, err := environment(info, cfg)
	if err != nil {
		return result, err
	}
	result["environment"] = env
	broker := pagebroker.Client{ControlSocketPath: cfg.PageBroker.ControlSocketPath}
	caps, err := broker.Capabilities(ctx)
	if err != nil {
		return result, err
	}
	cfg.CustomStorageAvailable = caps.CustomStorageAvailable
	result["custom_storage"] = caps.CustomStorageAvailable
	if r.Operation == "capture" {
		ready, err := snapshotruntime.ControlSentinelExists(pid, podcontract.ReadyForSnapshotFile)
		if err != nil {
			return result, err
		}
		if !ready {
			return result, errors.New("workload has not reached its clean capture boundary")
		}
		if err := claimTarget("/checkpoints/.attempts", r.Container); err != nil {
			return result, err
		}
		mutating = true
		artifact, err := executor.Checkpoint(ctx, rt, log, executor.CheckpointRequest{
			ContainerID: r.Container, ContainerName: "main", ContentUID: r.Artifact,
			StartedAt: time.Now(), NodeName: "standalone", Pod: env,
			CuInterposeRequested: r.Interpose,
		}, cfg)
		if err != nil {
			return result, err
		}
		// A published image is not accepted while a source task can still run.
		if err := rt.TerminateContainer(ctx, r.Container); err != nil {
			return result, err
		}
		result["published_artifact"] = artifact
		artifactPath, err := nsmount.ResolveArtifactPath(cfg.Storage.BasePath, r.Artifact, "main")
		if err != nil {
			return result, err
		}
		result["legacy_artifact_path"] = artifactPath
		result["source_terminated"] = true
		return result, nil
	}
	for _, sentinel := range []string{podcontract.RestoreCompleteFile, podcontract.RestoreFailedFile, podcontract.ReadyForSnapshotFile} {
		exists, err := snapshotruntime.ControlSentinelExists(pid, sentinel)
		if err != nil {
			return result, err
		}
		if exists {
			return result, fmt.Errorf("destination has stale %s; use a fresh container and control mount", sentinel)
		}
	}
	artifactPath, err := nsmount.ResolveArtifactPath(cfg.Storage.BasePath, r.Artifact, "main")
	if err != nil {
		return result, err
	}
	manifest, err := types.ReadManifest(artifactPath)
	if err != nil {
		return result, err
	}
	if manifest.Host.KernelVersion == "" || manifest.Host.CPUArch == "" || len(manifest.Host.CPUFeatures) == 0 {
		return result, errors.New("artifact lacks required host compatibility evidence")
	}
	if mismatch := compat.Compare(compat.GatePreflight, manifest.CompatEnvironment(), env); len(mismatch) != 0 {
		return result, compat.NewIncompatibleError(compat.GatePreflight, mismatch)
	}
	if manifest.K8s.CPULimit != env.CPULimit || manifest.K8s.MemoryLimit != env.MemoryLimit {
		return result, errors.New("experiment requires identical source/destination CPU and memory limits")
	}
	if err := claimTarget("/checkpoints/.attempts", r.Container); err != nil {
		return result, err
	}
	mutating = true
	restored, err := executor.Restore(ctx, rt, log, executor.RestoreRequest{
		ContentUID: r.Artifact, BasePath: cfg.Storage.BasePath, ContainerID: r.Container,
		StartedAt: time.Now(), ArtifactContainerName: "main", DestinationContainerName: "main",
		CustomStorageAvailable: cfg.CustomStorageAvailable, PageBrokerControlSocketPath: cfg.PageBroker.ControlSocketPath,
		PageBrokerRestoreMode: cfg.PageBroker.RestoreMode, WithoutProxy: true,
	}, nsmount.New(log))
	if err != nil {
		return result, err
	}
	// This releases the application's restore wait. It is NOT serving readiness.
	if err := snapshotruntime.WriteControlSentinel(restored.PlaceholderHostPID, podcontract.RestoreCompleteFile, podcontract.FormatRestoreComplete(podcontract.RestoreComplete{PID: restored.RestoredPID})); err != nil {
		return result, err
	}
	result["restored_pid"] = restored.RestoredPID
	result["memory_ready"] = true
	return result, nil
}

func main() {
	var r request
	decoder := json.NewDecoder(io.LimitReader(os.Stdin, 16384))
	decoder.DisallowUnknownFields()
	err := decoder.Decode(&r)
	if err == nil {
		var extra any
		if e := decoder.Decode(&extra); e != io.EOF {
			err = errors.New("request must contain exactly one JSON object")
		}
	}
	if err == nil {
		err = r.validate()
	}
	if err == nil && os.Geteuid() != 0 {
		err = errors.New("run only in the authorized privileged host-driver container")
	}
	result := map[string]any{"snapshot_revision": snapshotRevision, "status": "FAIL"}
	if err == nil {
		ctx, cancel := context.WithTimeout(context.Background(), time.Duration(r.TimeoutSeconds)*time.Second)
		defer cancel()
		var observed map[string]any
		observed, err = execute(ctx, r)
		for key, value := range observed {
			result[key] = value
		}
	}
	if err != nil {
		result["error"] = err.Error()
	} else {
		result["status"] = "PASS"
	}
	if e := json.NewEncoder(os.Stdout).Encode(result); e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
	if err != nil {
		os.Exit(1)
	}
}
