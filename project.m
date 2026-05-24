clc;
clear;
close all;

%% =====================================================
% AI-Assisted 5G Network Slicing using Q-Learning
%% =====================================================

%% PARAMETERS

episodes = 2000;

alpha = 0.1;
gamma = 0.9;
epsilon = 0.2;

numTrafficLevels = 3;

% Total states = 3 x 3 x 3 = 27
numStates = 27;

% Action Space
% [eMBB URLLC mMTC]

actions = [
    50 30 20;
    40 40 20;
    60 20 20;
    30 50 20;
    45 35 20;
    55 25 20;
    35 45 20;
    33 33 34
];

numActions = size(actions,1);

% Initialize Q Table
Q = zeros(numStates, numActions);

%% PERFORMANCE TRACKING

reward_history = zeros(episodes,1);
throughput_history = zeros(episodes,1);
delay_history = zeros(episodes,1);
packetloss_history = zeros(episodes,1);

%% =====================================================
% TRAINING LOOP
%% =====================================================

for episode = 1:episodes

    %% RANDOM TRAFFIC GENERATION

    eMBB_load = randi([10 100]);
    URLLC_load = randi([10 80]);
    mMTC_load = randi([10 60]);

    %% STATE ENCODING

    eMBB_state = trafficLevel(eMBB_load);
    URLLC_state = trafficLevel(URLLC_load);
    mMTC_state = trafficLevel(mMTC_load);

    state = stateEncoder(eMBB_state, URLLC_state, mMTC_state);

    %% EPSILON GREEDY POLICY

    if rand < epsilon
        action = randi(numActions);
    else
        [~, action] = max(Q(state,:));
    end

    allocation = actions(action,:);

    %% =================================================
    % NETWORK PERFORMANCE MODEL
    %% =================================================

    % Throughput
    throughput = ...
        min(eMBB_load, allocation(1)) + ...
        min(URLLC_load, allocation(2)) + ...
        min(mMTC_load, allocation(3));

    % Delay Model
    delay = ...
        abs(eMBB_load-allocation(1))*0.2 + ...
        abs(URLLC_load-allocation(2))*0.6 + ...
        abs(mMTC_load-allocation(3))*0.1;

    % Packet Loss
    overload = max(0, eMBB_load-allocation(1)) + ...
               max(0, URLLC_load-allocation(2)) + ...
               max(0, mMTC_load-allocation(3));

    packet_loss = overload * 0.3;

    % Slice Isolation Penalty
    isolation_penalty = ...
        abs(allocation(1)-eMBB_load)*0.05 + ...
        abs(allocation(2)-URLLC_load)*0.08;

    %% =================================================
    % REWARD FUNCTION
    %% =================================================

    reward = ...
        throughput ...
        - delay ...
        - packet_loss ...
        - isolation_penalty;

    %% NEXT STATE

    next_eMBB = eMBB_load + randi([-15 15]);
    next_URLLC = URLLC_load + randi([-10 10]);
    next_mMTC = mMTC_load + randi([-8 8]);

    next_eMBB = max(1,next_eMBB);
    next_URLLC = max(1,next_URLLC);
    next_mMTC = max(1,next_mMTC);

    next_state = stateEncoder( ...
        trafficLevel(next_eMBB), ...
        trafficLevel(next_URLLC), ...
        trafficLevel(next_mMTC));

    %% =================================================
    % Q LEARNING UPDATE
    %% =================================================

    Q(state,action) = Q(state,action) + ...
        alpha * (reward + ...
        gamma * max(Q(next_state,:)) - ...
        Q(state,action));

    %% STORE RESULTS

    reward_history(episode) = reward;
    throughput_history(episode) = throughput;
    delay_history(episode) = delay;
    packetloss_history(episode) = packet_loss;

end

%% =====================================================
% RESULTS
%% =====================================================

disp('====================================');
disp('FINAL Q TABLE');
disp('====================================');
disp(Q);

%% =====================================================
% PLOTS
%% =====================================================

% Reward Plot
figure;
plot(reward_history,'LineWidth',1.5);
xlabel('Episodes');
ylabel('Reward');
title('Reward vs Episodes');
grid on;

% Moving Average Reward
figure;
movingReward = movmean(reward_history,50);
plot(movingReward,'LineWidth',2);
xlabel('Episodes');
ylabel('Average Reward');
title('Convergence Analysis');
grid on;

% Throughput
figure;
plot(throughput_history,'LineWidth',1.5);
xlabel('Episodes');
ylabel('Throughput');
title('Throughput Performance');
grid on;

% Delay
figure;
plot(delay_history,'LineWidth',1.5);
xlabel('Episodes');
ylabel('Delay');
title('Delay Performance');
grid on;

% Packet Loss
figure;
plot(packetloss_history,'LineWidth',1.5);
xlabel('Episodes');
ylabel('Packet Loss');
title('Packet Loss Analysis');
grid on;

% Q Table Heatmap
figure;
imagesc(Q);
colorbar;
xlabel('Actions');
ylabel('States');
title('Q Table Heatmap');

%% =====================================================
% TESTING PHASE
%% =====================================================

disp('====================================');
disp('TESTING TRAINED AGENT');
disp('====================================');

for test = 1:5

    eMBB_load = randi([10 100]);
    URLLC_load = randi([10 80]);
    mMTC_load = randi([10 60]);

    state = stateEncoder( ...
        trafficLevel(eMBB_load), ...
        trafficLevel(URLLC_load), ...
        trafficLevel(mMTC_load));

    [~, best_action] = max(Q(state,:));

    allocation = actions(best_action,:);

    fprintf('\nTEST CASE %d\n',test);

    fprintf('Traffic Load:\n');
    fprintf('eMBB  = %d\n',eMBB_load);
    fprintf('URLLC = %d\n',URLLC_load);
    fprintf('mMTC  = %d\n',mMTC_load);

    fprintf('\nOptimal Allocation:\n');
    fprintf('eMBB  = %d\n',allocation(1));
    fprintf('URLLC = %d\n',allocation(2));
    fprintf('mMTC  = %d\n',allocation(3));

end

disp('====================================');
disp('SIMULATION COMPLETED');
disp('====================================');

%% =====================================================
% LOCAL FUNCTIONS
%% =====================================================

function level = trafficLevel(load)

    if load < 35
        level = 1;
    elseif load < 70
        level = 2;
    else
        level = 3;
    end

end

function state = stateEncoder(e,u,m)

    state = (e-1)*9 + (u-1)*3 + m;

end