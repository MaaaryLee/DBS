# ===== MATLAB Online integration helpers =====
import os
import time
from typing import Iterable, List, Optional
import glob

def get_sim_json_mtime(json_path: str = "matlab_data/simulation_params.json") -> Optional[float]:
    """
    Return the last modified time of the simulation params JSON, or None if it does not exist.
    """
    try:
        return os.path.getmtime(json_path)
    except FileNotFoundError:
        return None

def wait_for_json_change(
    json_path: str = "matlab_data/simulation_params.json",
    poll_seconds: float = 2.0,
    last_seen_mtime: Optional[float] = None,
    timeout_seconds: Optional[float] = None,
) -> Optional[float]:
    """
    Block until simulation_params.json changes on disk, then return the new mtime.
    Use this to know when Python has produced a new request for MATLAB.
    Returns None if the timeout is reached without change.
    """
    start = time.time()
    if last_seen_mtime is None:
        last_seen_mtime = get_sim_json_mtime(json_path)
    while True:
        current = get_sim_json_mtime(json_path)
        if current is not None and current != last_seen_mtime:
            return current
        if timeout_seconds is not None and (time.time() - start) >= timeout_seconds:
            return None
        time.sleep(poll_seconds)

def print_matlab_reminder(json_path: str = "matlab_data/simulation_params.json") -> None:
    """
    Print a concise operator checklist telling the user what to do in MATLAB Online.
    """
    print("📋 New parameters detected in", json_path)
    print("1) Upload simulation_params.json to MATLAB Online into matlab_data")
    print("2) In MATLAB, run: run_simulation_online")
    print("3) Download simulation_results.mat back into matlab_data locally")
    print("4) Continue training in Python")

def list_mat_results(
    folder: str = "matlab_data",
    prefix: str = "simulation_results",
    allowed_suffixes: Iterable[str] = (".mat",),
    sort_by: str = "name",  # 'name' or 'mtime'
    reverse: bool = False,
) -> List[str]:
    """
    List result MAT files like simulation_results.mat, simulation_results_001.mat, etc.
    sort_by='name' sorts lexicographically. sort_by='mtime' sorts by modification time.
    """
    paths: List[str] = []
    for ext in allowed_suffixes:
        paths.extend(glob.glob(os.path.join(folder, f"{prefix}*{ext}")))
    if sort_by == "mtime":
        paths.sort(key=lambda p: os.path.getmtime(p), reverse=reverse)
    else:
        paths.sort(reverse=reverse)
    return paths

function run_simulation_online()
% MATLAB Online script for DBS simulation
% This script reads parameters from simulation_params.json and runs the simulation
% Ensure required code is on the path
base = fileparts(mfilename('fullpath'));      % matlab_data
repo = fileparts(base);                       % parent that holds bgn_init.m and bgn_step.m
addpath(repo);

% Handle both layouts, gating next to matlab_data, or nested gating/gating from zip
candidates = { fullfile(repo,'gating'), fullfile(repo,'gating','gating') };
added = false;
for c = candidates
    if exist(c{1},'dir')
        addpath(genpath(c{1}));
        added = true;
        break
    end
end
if ~added
    error('Could not find the gating folder. Place it next to matlab_data or update the path above.');
end

try
    % Read simulation parameters
    params_file = 'simulation_params.json';
    if ~exist(params_file, 'file')
        error('simulation_params.json not found. Please upload it first.');
    end
    
    % Read JSON parameters (requires JSON toolbox or manual parsing)
    fid = fopen(params_file, 'r');
    raw = fread(fid, inf);
    str = char(raw');
    fclose(fid);
    
    % Parse JSON manually (simple approach)
    freq = extract_value(str, '"freq"');
    amp = extract_value(str, '"amp"');
    pd = extract_value(str, '"pd"');
    tmax = extract_value(str, '"tmax"');
    sim_time = extract_value(str, '"sim_time"');
    
    fprintf('Running simulation with: freq=%.1fHz, amp=%.1fmA, pd=%d, tmax=%dms\n', ...
            freq, amp, pd, tmax);
    
    % Initialize simulation
    bgn_init(pd, tmax);
    
    % Run simulation step
    [terminated, sgis] = bgn_step(freq, amp, sim_time);
    
    % Save results
    save('simulation_results.mat', 'terminated', 'sgis', 'freq', 'amp', 'pd', 'tmax');
    
    fprintf('✅ Simulation completed. Results saved to simulation_results.mat\n');
    fprintf('Terminated: %d\n', terminated);
    
catch ME
    fprintf('❌ Error: %s\n', ME.message);
    % Save error information
    error_info = struct('error', ME.message, 'stack', ME.stack);
    save('simulation_error.mat', 'error_info');
end
end

function value = extract_value(json_str, field_name)
    % Simple JSON value extraction
    pattern = [field_name, '\s*:\s*([^,}\]]+)'];
    match = regexp(json_str, pattern, 'tokens');
    if ~isempty(match)
        value_str = strtrim(match{1}{1});
        % Remove quotes if present
        if value_str(1) == '"' && value_str(end) == '"'
            value_str = value_str(2:end-1);
        end
        value = str2double(value_str);
        if isnan(value)
            value = value_str;
        end
    else
        value = [];
    end
end