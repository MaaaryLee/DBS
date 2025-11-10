function run_simulation_online()
    % MATLAB Online script for DBS simulation
    % This script reads parameters from simulation_params.json and runs the simulation
    
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
