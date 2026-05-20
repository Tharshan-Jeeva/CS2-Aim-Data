#include <sourcemod>
#include <sdktools>

#pragma semicolon 1
#pragma newdecls required

#define PLUGIN_VERSION "0.1.0"

public Plugin myinfo = {
    name = "CS Aim Live Controller",
    author = "Tharshan-Jeeva",
    description = "SourceMod-native low-latency aim controller (raw/smooth/humanised)",
    version = PLUGIN_VERSION,
    url = ""
};

enum AimMode
{
    AimMode_Raw = 0,
    AimMode_Smooth = 1,
    AimMode_Humanised = 2,
    AimMode_HumanisedHigh = 3
};

ConVar g_cvEnable;
ConVar g_cvFov;
ConVar g_cvTargetZOffset;
ConVar g_cvLateralOffset;
ConVar g_cvSmoothGain;
ConVar g_cvHumanGain;
ConVar g_cvReactionMs;
ConVar g_cvJitterDeg;
ConVar g_cvLockMs;
ConVar g_cvLostGraceMs;
ConVar g_cvSwitchImprovement;
ConVar g_cvPredictEnabled;
ConVar g_cvLeadSeconds;
ConVar g_cvMaxLeadUnits;
ConVar g_cvPunchCompensate;
ConVar g_cvDebug;

// humanised_high tuning
ConVar g_cvHHReactionMs;
ConVar g_cvHHGainFar;
ConVar g_cvHHGainNear;
ConVar g_cvHHDecelDeg;
ConVar g_cvHHJitterMoving;
ConVar g_cvHHJitterSettled;
ConVar g_cvHHOvershootProb;
ConVar g_cvHHOvershootMult;
ConVar g_cvHHDriftDeg;
ConVar g_cvHHDriftHz;
ConVar g_cvHHSettleDeg;

bool  g_bActive[MAXPLAYERS + 1];
int   g_iMode[MAXPLAYERS + 1];
int   g_iTarget[MAXPLAYERS + 1];
float g_flEngageStartTime[MAXPLAYERS + 1];
float g_flEngageStartYaw[MAXPLAYERS + 1];
float g_flEngageStartPitch[MAXPLAYERS + 1];
float g_flReactionReadyTime[MAXPLAYERS + 1];
float g_flLastTargetSeenTime[MAXPLAYERS + 1];
float g_flLastAppliedYaw[MAXPLAYERS + 1];
float g_flLastAppliedPitch[MAXPLAYERS + 1];
float g_flLastAngularDistance[MAXPLAYERS + 1];

// humanised_high per-client state
bool  g_bHHOvershootActive[MAXPLAYERS + 1];
float g_flHHDriftPhaseYaw[MAXPLAYERS + 1];
float g_flHHDriftPhasePitch[MAXPLAYERS + 1];

public void OnPluginStart()
{
    g_cvEnable = CreateConVar("sm_nativeaim_enable", "1",
        "Global enable for the SourceMod-native aim controller (0=off, 1=on)",
        FCVAR_NONE, true, 0.0, true, 1.0);
    g_cvFov = CreateConVar("sm_nativeaim_fov", "35.0",
        "Max angular distance (degrees) for target acquisition",
        FCVAR_NONE, true, 1.0, true, 180.0);
    g_cvTargetZOffset = CreateConVar("sm_nativeaim_target_z_offset", "-8.0",
        "Vertical offset from target eye position (negative = lower, toward neck/chest)");
    g_cvLateralOffset = CreateConVar("sm_nativeaim_lateral_offset", "0.0",
        "World-space lateral offset in units (positive = left of aimer's view)");
    g_cvSmoothGain = CreateConVar("sm_nativeaim_smooth_gain", "0.45",
        "Smooth mode per-tick correction gain (0.0-1.0)",
        FCVAR_NONE, true, 0.0, true, 1.0);
    g_cvHumanGain = CreateConVar("sm_nativeaim_human_gain", "0.35",
        "Humanised mode tracking gain after reaction (0.0-1.0)",
        FCVAR_NONE, true, 0.0, true, 1.0);
    g_cvReactionMs = CreateConVar("sm_nativeaim_reaction_ms", "80.0",
        "Humanised mode reaction delay in milliseconds",
        FCVAR_NONE, true, 0.0, true, 1000.0);
    g_cvJitterDeg = CreateConVar("sm_nativeaim_jitter_deg", "0.12",
        "Humanised mode jitter amplitude in degrees",
        FCVAR_NONE, true, 0.0, true, 5.0);
    g_cvLockMs = CreateConVar("sm_nativeaim_lock_ms", "100.0",
        "Minimum time on a target before considering switches (ms)",
        FCVAR_NONE, true, 0.0, true, 5000.0);
    g_cvLostGraceMs = CreateConVar("sm_nativeaim_lost_grace_ms", "150.0",
        "Grace period (ms) before dropping a target that briefly fails visibility",
        FCVAR_NONE, true, 0.0, true, 5000.0);
    g_cvSwitchImprovement = CreateConVar("sm_nativeaim_switch_improvement", "0.90",
        "Candidate must be this fraction of current angular distance to switch",
        FCVAR_NONE, true, 0.1, true, 1.0);
    // Prediction is OFF by default for SM-native: the aim is applied in the
    // same server tick as the position read, so there is no round-trip latency
    // to compensate for. Leading only causes overshoot — especially when the
    // enemy changes direction (the predicted point flips ahead of velocity).
    g_cvPredictEnabled = CreateConVar("sm_nativeaim_predict_enabled", "0",
        "Lead moving targets using server-side velocity (0=off, 1=on)",
        FCVAR_NONE, true, 0.0, true, 1.0);
    g_cvLeadSeconds = CreateConVar("sm_nativeaim_lead_seconds", "0.020",
        "Prediction lead in seconds (XY only, ~2 ticks at 100Hz)",
        FCVAR_NONE, true, 0.0, true, 0.200);
    g_cvMaxLeadUnits = CreateConVar("sm_nativeaim_max_lead_units", "64.0",
        "Maximum lead distance in world units (prevents wild overshoot)",
        FCVAR_NONE, true, 0.0, true, 512.0);
    g_cvPunchCompensate = CreateConVar("sm_nativeaim_punch_compensate", "1",
        "Compensate for weapon recoil (punchangle) when applying view angles",
        FCVAR_NONE, true, 0.0, true, 1.0);
    g_cvDebug = CreateConVar("sm_nativeaim_debug", "0",
        "Print extra debug information",
        FCVAR_NONE, true, 0.0, true, 1.0);

    // ------- humanised_high (very-human) tuning -------
    g_cvHHReactionMs = CreateConVar("sm_nativeaim_hh_reaction_ms", "170.0",
        "humanised_high reaction delay before tracking begins (ms)",
        FCVAR_NONE, true, 0.0, true, 1000.0);
    g_cvHHGainFar = CreateConVar("sm_nativeaim_hh_gain_far", "0.55",
        "humanised_high per-tick gain when far from target (flick phase)",
        FCVAR_NONE, true, 0.0, true, 1.0);
    g_cvHHGainNear = CreateConVar("sm_nativeaim_hh_gain_near", "0.14",
        "humanised_high per-tick gain when near target (settle phase)",
        FCVAR_NONE, true, 0.0, true, 1.0);
    g_cvHHDecelDeg = CreateConVar("sm_nativeaim_hh_decel_deg", "8.0",
        "Angular distance (deg) above which gain_far is used; below, blends to gain_near",
        FCVAR_NONE, true, 0.5, true, 60.0);
    g_cvHHJitterMoving = CreateConVar("sm_nativeaim_hh_jitter_moving_deg", "0.55",
        "humanised_high jitter (deg) while crosshair is moving toward target",
        FCVAR_NONE, true, 0.0, true, 5.0);
    g_cvHHJitterSettled = CreateConVar("sm_nativeaim_hh_jitter_settled_deg", "0.07",
        "humanised_high jitter (deg) once crosshair is on target",
        FCVAR_NONE, true, 0.0, true, 5.0);
    g_cvHHOvershootProb = CreateConVar("sm_nativeaim_hh_overshoot_prob", "0.35",
        "Probability the flick overshoots when first acquiring a target",
        FCVAR_NONE, true, 0.0, true, 1.0);
    g_cvHHOvershootMult = CreateConVar("sm_nativeaim_hh_overshoot_mult", "1.45",
        "Gain multiplier while overshoot is active",
        FCVAR_NONE, true, 1.0, true, 3.0);
    g_cvHHDriftDeg = CreateConVar("sm_nativeaim_hh_drift_deg", "0.22",
        "Slow drift amplitude (deg) added when settled on target",
        FCVAR_NONE, true, 0.0, true, 2.0);
    g_cvHHDriftHz = CreateConVar("sm_nativeaim_hh_drift_hz", "1.3",
        "Drift oscillation frequency (Hz)",
        FCVAR_NONE, true, 0.05, true, 10.0);
    g_cvHHSettleDeg = CreateConVar("sm_nativeaim_hh_settle_deg", "1.5",
        "Angular distance (deg) at which the engagement is considered settled (clears overshoot)",
        FCVAR_NONE, true, 0.1, true, 10.0);

    RegConsoleCmd("sm_nativeaim_active", Cmd_SetActive,
        "Enable/disable native aim assist for yourself (0/1)");
    RegConsoleCmd("sm_nativeaim_mode", Cmd_SetMode,
        "Set native aim mode (raw/smooth/humanised/humanised_high)");
    RegConsoleCmd("sm_nativeaim_status", Cmd_Status,
        "Show native aim status for yourself");
    RegConsoleCmd("sm_nativeaim_me", Cmd_Me,
        "Print client debug info (index, name, team, alive, eye, angles)");

    for (int i = 0; i <= MAXPLAYERS; i++)
    {
        ResetClientState(i);
    }
}

public void OnClientPutInServer(int client)
{
    ResetClientState(client);
}

public void OnClientDisconnect(int client)
{
    ResetClientState(client);
}

void ResetClientState(int client)
{
    g_bActive[client] = false;
    g_iMode[client] = view_as<int>(AimMode_Raw);
    g_iTarget[client] = -1;
    g_flEngageStartTime[client] = 0.0;
    g_flEngageStartYaw[client] = 0.0;
    g_flEngageStartPitch[client] = 0.0;
    g_flReactionReadyTime[client] = 0.0;
    g_flLastTargetSeenTime[client] = 0.0;
    g_flLastAppliedYaw[client] = 0.0;
    g_flLastAppliedPitch[client] = 0.0;
    g_flLastAngularDistance[client] = 0.0;
    g_bHHOvershootActive[client] = false;
    g_flHHDriftPhaseYaw[client] = 0.0;
    g_flHHDriftPhasePitch[client] = 0.0;
}

// ----------------------------------------------------------------------------
// Commands
// ----------------------------------------------------------------------------

public Action Cmd_SetActive(int client, int args)
{
    if (client <= 0 || !IsClientInGame(client))
    {
        ReplyToCommand(client, "[NativeAim] Must be run by a player.");
        return Plugin_Handled;
    }

    if (args < 1)
    {
        ReplyToCommand(client, "[NativeAim] Active: %d", g_bActive[client] ? 1 : 0);
        return Plugin_Handled;
    }

    char arg[8];
    GetCmdArg(1, arg, sizeof(arg));
    bool active = StringToInt(arg) != 0;
    g_bActive[client] = active;
    if (!active)
    {
        g_iTarget[client] = -1;
    }
    ReplyToCommand(client, "[NativeAim] Active: %d", active ? 1 : 0);
    return Plugin_Handled;
}

public Action Cmd_SetMode(int client, int args)
{
    if (client <= 0 || !IsClientInGame(client))
    {
        ReplyToCommand(client, "[NativeAim] Must be run by a player.");
        return Plugin_Handled;
    }

    if (args < 1)
    {
        ReplyToCommand(client, "[NativeAim] Mode: %s", ModeName(g_iMode[client]));
        return Plugin_Handled;
    }

    char arg[16];
    GetCmdArg(1, arg, sizeof(arg));
    if (StrEqual(arg, "raw", false))
    {
        g_iMode[client] = view_as<int>(AimMode_Raw);
    }
    else if (StrEqual(arg, "smooth", false))
    {
        g_iMode[client] = view_as<int>(AimMode_Smooth);
    }
    else if (StrEqual(arg, "humanised_high", false)
          || StrEqual(arg, "humanized_high", false)
          || StrEqual(arg, "hh", false))
    {
        g_iMode[client] = view_as<int>(AimMode_HumanisedHigh);
    }
    else if (StrEqual(arg, "humanised", false) || StrEqual(arg, "humanized", false))
    {
        g_iMode[client] = view_as<int>(AimMode_Humanised);
    }
    else
    {
        ReplyToCommand(client, "[NativeAim] Unknown mode '%s' (use raw/smooth/humanised/humanised_high)", arg);
        return Plugin_Handled;
    }
    // Reset engagement so the new mode starts cleanly.
    g_iTarget[client] = -1;
    ReplyToCommand(client, "[NativeAim] Mode: %s", ModeName(g_iMode[client]));
    return Plugin_Handled;
}

public Action Cmd_Status(int client, int args)
{
    if (client <= 0 || !IsClientInGame(client))
    {
        ReplyToCommand(client, "[NativeAim] Must be run by a player.");
        return Plugin_Handled;
    }

    int target = g_iTarget[client];
    char targetName[64];
    if (target > 0 && IsClientInGame(target))
    {
        Format(targetName, sizeof(targetName), "%N (slot %d)", target, target);
    }
    else
    {
        strcopy(targetName, sizeof(targetName), "<none>");
    }

    ReplyToCommand(client,
        "[NativeAim] active=%d mode=%s target=%s fov=%.1f z_offset=%.1f lateral=%.1f smooth=%.2f human=%.2f last_dist=%.2f last_yaw=%.2f last_pitch=%.2f",
        g_bActive[client] ? 1 : 0,
        ModeName(g_iMode[client]),
        targetName,
        g_cvFov.FloatValue,
        g_cvTargetZOffset.FloatValue,
        g_cvLateralOffset.FloatValue,
        g_cvSmoothGain.FloatValue,
        g_cvHumanGain.FloatValue,
        g_flLastAngularDistance[client],
        g_flLastAppliedYaw[client],
        g_flLastAppliedPitch[client]);
    return Plugin_Handled;
}

public Action Cmd_Me(int client, int args)
{
    if (client <= 0 || !IsClientInGame(client))
    {
        ReplyToCommand(client, "[NativeAim] Must be run by a player.");
        return Plugin_Handled;
    }

    float eyePos[3], eyeAngles[3];
    GetClientEyePosition(client, eyePos);
    GetClientEyeAngles(client, eyeAngles);
    bool alive = IsPlayerAlive(client);
    int team = GetClientTeam(client);

    ReplyToCommand(client,
        "[NativeAim] slot=%d name=%N team=%d alive=%d eye=[%.1f,%.1f,%.1f] angles=[pitch=%.2f yaw=%.2f]",
        client, client, team, alive ? 1 : 0,
        eyePos[0], eyePos[1], eyePos[2],
        eyeAngles[0], eyeAngles[1]);
    return Plugin_Handled;
}

char ModeName_buf[16];
char[] ModeName(int mode)
{
    if (mode == view_as<int>(AimMode_Raw))                 strcopy(ModeName_buf, sizeof(ModeName_buf), "raw");
    else if (mode == view_as<int>(AimMode_Smooth))         strcopy(ModeName_buf, sizeof(ModeName_buf), "smooth");
    else if (mode == view_as<int>(AimMode_Humanised))      strcopy(ModeName_buf, sizeof(ModeName_buf), "humanised");
    else if (mode == view_as<int>(AimMode_HumanisedHigh))  strcopy(ModeName_buf, sizeof(ModeName_buf), "humanised_high");
    else strcopy(ModeName_buf, sizeof(ModeName_buf), "?");
    return ModeName_buf;
}

// ----------------------------------------------------------------------------
// Angle math
// ----------------------------------------------------------------------------

float NormalizeYaw(float yaw)
{
    while (yaw > 180.0)  yaw -= 360.0;
    while (yaw < -180.0) yaw += 360.0;
    return yaw;
}

float ClampPitch(float pitch)
{
    if (pitch > 89.0)  return 89.0;
    if (pitch < -89.0) return -89.0;
    return pitch;
}

float AngleDelta(float fromAngle, float toAngle)
{
    float delta = toAngle - fromAngle;
    while (delta > 180.0)  delta -= 360.0;
    while (delta < -180.0) delta += 360.0;
    return delta;
}

void ComputeAngleToPoint(float eyePos[3], float targetPoint[3], float outAngles[3])
{
    float dx = targetPoint[0] - eyePos[0];
    float dy = targetPoint[1] - eyePos[1];
    float dz = targetPoint[2] - eyePos[2];
    float distH = SquareRoot(dx * dx + dy * dy);

    float yaw = RadToDeg(ArcTangent2(dy, dx));
    float pitch = -RadToDeg(ArcTangent2(dz, distH));

    outAngles[0] = ClampPitch(pitch);
    outAngles[1] = NormalizeYaw(yaw);
    outAngles[2] = 0.0;
}

float GetAngularDistance(float currentYaw, float currentPitch, float targetYaw, float targetPitch)
{
    float dyaw = AngleDelta(currentYaw, targetYaw);
    float dpitch = AngleDelta(currentPitch, targetPitch);
    return SquareRoot(dyaw * dyaw + dpitch * dpitch);
}

// ----------------------------------------------------------------------------
// Target selection
// ----------------------------------------------------------------------------

bool IsValidEnemy(int client, int target)
{
    if (target <= 0 || target > MaxClients) return false;
    if (target == client) return false;
    if (!IsClientInGame(target)) return false;
    if (!IsPlayerAlive(target)) return false;
    if (GetClientTeam(target) == GetClientTeam(client)) return false;
    return true;
}

bool GetAimPoint(int client, int target, float aimPoint[3])
{
    if (!IsValidEnemy(client, target)) return false;

    GetClientEyePosition(target, aimPoint);
    aimPoint[2] += g_cvTargetZOffset.FloatValue;

    // XY velocity-based lead. Z is intentionally NOT predicted: jump/crouch
    // toggles produce large Z swings that ruin the aim if leaded.
    if (g_cvPredictEnabled.BoolValue)
    {
        float vel[3];
        GetEntPropVector(target, Prop_Data, "m_vecVelocity", vel);
        float lead = g_cvLeadSeconds.FloatValue;
        float dx = vel[0] * lead;
        float dy = vel[1] * lead;
        float mag = SquareRoot(dx * dx + dy * dy);
        float maxLead = g_cvMaxLeadUnits.FloatValue;
        if (mag > maxLead && mag > 0.001)
        {
            float scale = maxLead / mag;
            dx *= scale;
            dy *= scale;
        }
        aimPoint[0] += dx;
        aimPoint[1] += dy;
    }

    float lateral = g_cvLateralOffset.FloatValue;
    if (lateral != 0.0)
    {
        float eyePos[3];
        GetClientEyePosition(client, eyePos);
        float dx = aimPoint[0] - eyePos[0];
        float dy = aimPoint[1] - eyePos[1];
        float distH = SquareRoot(dx * dx + dy * dy);
        if (distH > 0.001)
        {
            float leftX = -dy / distH;
            float leftY =  dx / distH;
            aimPoint[0] += leftX * lateral;
            aimPoint[1] += leftY * lateral;
        }
    }
    return true;
}

public bool TraceFilter_IgnoreClient(int entity, int contentsMask, any data)
{
    return entity != view_as<int>(data);
}

bool IsVisibleToClient(int client, int target, float aimPoint[3])
{
    float eyePos[3];
    GetClientEyePosition(client, eyePos);

    Handle trace = TR_TraceRayFilterEx(eyePos, aimPoint, MASK_SHOT,
        RayType_EndPoint, TraceFilter_IgnoreClient, client);
    if (trace == INVALID_HANDLE) return true;

    // Visible only if the first thing the ray hits IS the target entity.
    // Hitting world (entity 0/-1), another player, or a prop = blocked.
    bool visible;
    if (TR_DidHit(trace))
    {
        int hit = TR_GetEntityIndex(trace);
        visible = (hit == target);
    }
    else
    {
        // Clear path to aim point — target is visible.
        visible = true;
    }
    delete trace;
    return visible;
}

int FindBestTarget(int client, float currentYaw, float currentPitch, float eyePos[3], float &outDist)
{
    int best = -1;
    float bestDist = g_cvFov.FloatValue;
    outDist = bestDist;

    for (int i = 1; i <= MaxClients; i++)
    {
        if (!IsValidEnemy(client, i)) continue;

        float aimPoint[3];
        if (!GetAimPoint(client, i, aimPoint)) continue;

        float angles[3];
        ComputeAngleToPoint(eyePos, aimPoint, angles);
        float dist = GetAngularDistance(currentYaw, currentPitch, angles[1], angles[0]);
        if (dist > g_cvFov.FloatValue) continue;

        if (!IsVisibleToClient(client, i, aimPoint)) continue;

        if (dist < bestDist)
        {
            bestDist = dist;
            best = i;
        }
    }

    outDist = bestDist;
    return best;
}

// ----------------------------------------------------------------------------
// Per-tick aim
// ----------------------------------------------------------------------------

public Action OnPlayerRunCmd(int client, int &buttons, int &impulse,
    float vel[3], float angles[3], int &weapon,
    int &subtype, int &cmdnum, int &tickcount, int &seed, int mouse[2])
{
    if (!g_cvEnable.BoolValue) return Plugin_Continue;
    if (client <= 0 || client > MaxClients) return Plugin_Continue;
    if (!IsClientInGame(client) || !IsPlayerAlive(client)) return Plugin_Continue;
    if (!g_bActive[client]) return Plugin_Continue;

    float eyePos[3];
    GetClientEyePosition(client, eyePos);

    float currentPitch = angles[0];
    float currentYaw = angles[1];

    float now = GetGameTime();
    float lockSec   = g_cvLockMs.FloatValue / 1000.0;
    float graceSec  = g_cvLostGraceMs.FloatValue / 1000.0;
    float switchFrac = g_cvSwitchImprovement.FloatValue;

    int currentTarget = g_iTarget[client];
    bool keepCurrent = false;
    float currentDist = 0.0;
    float currentAimPoint[3];
    float currentAimAngles[3];

    if (currentTarget > 0 && IsValidEnemy(client, currentTarget)
        && GetAimPoint(client, currentTarget, currentAimPoint))
    {
        ComputeAngleToPoint(eyePos, currentAimPoint, currentAimAngles);
        currentDist = GetAngularDistance(currentYaw, currentPitch,
            currentAimAngles[1], currentAimAngles[0]);

        bool visible = IsVisibleToClient(client, currentTarget, currentAimPoint);
        if (visible)
        {
            g_flLastTargetSeenTime[client] = now;
        }
        bool withinGrace = (now - g_flLastTargetSeenTime[client]) <= graceSec;

        if (currentDist <= g_cvFov.FloatValue && (visible || withinGrace))
        {
            keepCurrent = true;
        }
    }

    // Always scan for the closest-to-crosshair candidate. Lock window only
    // gates whether we *switch* to it; the scan itself runs every tick so we
    // can react instantly when the crosshair moves onto a closer enemy.
    float candidateDist = g_cvFov.FloatValue;
    int candidate = FindBestTarget(client, currentYaw, currentPitch, eyePos, candidateDist);
    bool locked = keepCurrent && (now - g_flEngageStartTime[client]) < lockSec;

    int chosen = -1;
    if (keepCurrent && candidate > 0 && candidate != currentTarget)
    {
        // Switch only if locked window has passed AND candidate is meaningfully
        // closer to the crosshair than the current target.
        if (!locked && candidateDist < currentDist * switchFrac)
        {
            chosen = candidate;
        }
        else
        {
            chosen = currentTarget;
        }
    }
    else if (keepCurrent)
    {
        chosen = currentTarget;
    }
    else
    {
        chosen = candidate;
    }

    if (chosen <= 0)
    {
        g_iTarget[client] = -1;
        return Plugin_Continue;
    }

    // New engagement -> reset engage state.
    if (chosen != g_iTarget[client])
    {
        g_iTarget[client] = chosen;
        g_flEngageStartTime[client] = now;
        g_flEngageStartYaw[client] = currentYaw;
        g_flEngageStartPitch[client] = currentPitch;
        float reactionMs = g_cvReactionMs.FloatValue;
        if (g_iMode[client] == view_as<int>(AimMode_HumanisedHigh))
        {
            reactionMs = g_cvHHReactionMs.FloatValue;
            // ±20% reaction jitter so flicks don't all start on the same tick.
            reactionMs *= GetRandomFloat(0.8, 1.2);
            g_bHHOvershootActive[client] =
                GetRandomFloat(0.0, 1.0) < g_cvHHOvershootProb.FloatValue;
            g_flHHDriftPhaseYaw[client]   = GetRandomFloat(0.0, 6.2831853);
            g_flHHDriftPhasePitch[client] = GetRandomFloat(0.0, 6.2831853);
        }
        g_flReactionReadyTime[client] = now + (reactionMs / 1000.0);
        g_flLastTargetSeenTime[client] = now;
    }

    float aimPoint[3];
    if (!GetAimPoint(client, chosen, aimPoint))
    {
        return Plugin_Continue;
    }

    float targetAngles[3];
    ComputeAngleToPoint(eyePos, aimPoint, targetAngles);
    float targetPitch = targetAngles[0];
    float targetYaw   = targetAngles[1];

    g_flLastAngularDistance[client] = GetAngularDistance(
        currentYaw, currentPitch, targetYaw, targetPitch);

    int mode = g_iMode[client];
    float desiredYaw = currentYaw;
    float desiredPitch = currentPitch;

    if (mode == view_as<int>(AimMode_Raw))
    {
        desiredYaw = targetYaw;
        desiredPitch = targetPitch;
    }
    else if (mode == view_as<int>(AimMode_Smooth))
    {
        float gain = g_cvSmoothGain.FloatValue;
        float dyaw = AngleDelta(currentYaw, targetYaw);
        float dpitch = AngleDelta(currentPitch, targetPitch);
        desiredYaw = NormalizeYaw(currentYaw + dyaw * gain);
        desiredPitch = ClampPitch(currentPitch + dpitch * gain);
    }
    else if (mode == view_as<int>(AimMode_Humanised))
    {
        if (now < g_flReactionReadyTime[client])
        {
            // Still in reaction window — leave aim untouched.
            return Plugin_Continue;
        }
        float gain = g_cvHumanGain.FloatValue;
        float dyaw = AngleDelta(currentYaw, targetYaw);
        float dpitch = AngleDelta(currentPitch, targetPitch);
        desiredYaw = currentYaw + dyaw * gain;
        desiredPitch = currentPitch + dpitch * gain;

        float jitter = g_cvJitterDeg.FloatValue;
        if (jitter > 0.0)
        {
            desiredYaw   += GetRandomFloat(-jitter, jitter);
            desiredPitch += GetRandomFloat(-jitter * 0.6, jitter * 0.6);
        }
        desiredYaw = NormalizeYaw(desiredYaw);
        desiredPitch = ClampPitch(desiredPitch);
    }
    else // HumanisedHigh — flick-then-settle with distance-scaled jitter + drift
    {
        if (now < g_flReactionReadyTime[client])
        {
            return Plugin_Continue;
        }

        float angDist = g_flLastAngularDistance[client];
        float decel = g_cvHHDecelDeg.FloatValue;
        // t=1 when far (>=decel), t=0 when on target. Drives gain + jitter.
        float t = angDist / decel;
        if (t > 1.0) t = 1.0;
        if (t < 0.0) t = 0.0;

        float gainFar  = g_cvHHGainFar.FloatValue;
        float gainNear = g_cvHHGainNear.FloatValue;
        float gain = gainNear + (gainFar - gainNear) * t;

        if (g_bHHOvershootActive[client])
        {
            gain *= g_cvHHOvershootMult.FloatValue;
            // Clear once we've crossed close enough — the next tick will pull back.
            if (angDist < g_cvHHSettleDeg.FloatValue)
            {
                g_bHHOvershootActive[client] = false;
            }
        }
        if (gain > 1.0) gain = 1.0;

        float dyaw = AngleDelta(currentYaw, targetYaw);
        float dpitch = AngleDelta(currentPitch, targetPitch);
        desiredYaw   = currentYaw   + dyaw   * gain;
        desiredPitch = currentPitch + dpitch * gain;

        // Distance-scaled jitter: big shake during flick, fine tremor when settled.
        float jMove = g_cvHHJitterMoving.FloatValue;
        float jRest = g_cvHHJitterSettled.FloatValue;
        float jitter = jRest + (jMove - jRest) * t;
        if (jitter > 0.0)
        {
            desiredYaw   += GetRandomFloat(-jitter, jitter);
            desiredPitch += GetRandomFloat(-jitter * 0.7, jitter * 0.7);
        }

        // Slow drift, strongest when on target (humans wobble even when still).
        float driftAmp = g_cvHHDriftDeg.FloatValue * (1.0 - t);
        if (driftAmp > 0.0)
        {
            float w = g_cvHHDriftHz.FloatValue * 6.2831853;
            float tnow = GetEngineTime();
            desiredYaw   += driftAmp        * Sine(tnow * w + g_flHHDriftPhaseYaw[client]);
            desiredPitch += driftAmp * 0.6  * Sine(tnow * w * 1.27 + g_flHHDriftPhasePitch[client]);
        }

        desiredYaw = NormalizeYaw(desiredYaw);
        desiredPitch = ClampPitch(desiredPitch);
    }

    // Recoil compensation: GetClientEyeAngles() returns base + punchangle, but
    // when we write angles[] the engine adds punchangle on top again. Subtract
    // the current punchangle so the final aim lands exactly on the target
    // regardless of recoil state (matches cs_aim_override.sp behaviour).
    float applyPitch = desiredPitch;
    float applyYaw = desiredYaw;
    if (g_cvPunchCompensate.BoolValue
        && HasEntProp(client, Prop_Send, "m_vecPunchAngle"))
    {
        float punchAngle[3];
        GetEntPropVector(client, Prop_Send, "m_vecPunchAngle", punchAngle);
        applyPitch -= punchAngle[0];
        applyYaw   -= punchAngle[1];
    }

    angles[0] = applyPitch;
    angles[1] = applyYaw;
    angles[2] = 0.0;

    float applyAngles[3];
    applyAngles[0] = applyPitch;
    applyAngles[1] = applyYaw;
    applyAngles[2] = 0.0;
    TeleportEntity(client, NULL_VECTOR, applyAngles, NULL_VECTOR);

    g_flLastAppliedYaw[client] = desiredYaw;
    g_flLastAppliedPitch[client] = desiredPitch;

    if (g_cvDebug.BoolValue)
    {
        static int s_DebugCount = 0;
        s_DebugCount++;
        if (s_DebugCount % 50 == 0)
        {
            PrintToServer("[NativeAim] %N mode=%s target=%d dist=%.2f -> yaw=%.2f pitch=%.2f",
                client, ModeName(mode), chosen,
                g_flLastAngularDistance[client], desiredYaw, desiredPitch);
        }
    }

    return Plugin_Changed;
}
