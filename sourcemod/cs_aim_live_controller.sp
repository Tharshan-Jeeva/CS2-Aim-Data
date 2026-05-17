#include <sourcemod>
#include <sdktools>

#pragma semicolon 1
#pragma newdecls required

#define PLUGIN_VERSION "0.1.0"

public Plugin myinfo = {
    name = "CS Aim Live Controller",
    author = "Tzan",
    description = "SourceMod-native low-latency aim controller (raw/smooth/humanised)",
    version = PLUGIN_VERSION,
    url = ""
};

enum AimMode
{
    AimMode_Raw = 0,
    AimMode_Smooth = 1,
    AimMode_Humanised = 2
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
ConVar g_cvDebug;

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

public void OnPluginStart()
{
    g_cvEnable = CreateConVar("sm_nativeaim_enable", "1",
        "Global enable for the SourceMod-native aim controller (0=off, 1=on)",
        FCVAR_NONE, true, 0.0, true, 1.0);
    g_cvFov = CreateConVar("sm_nativeaim_fov", "35.0",
        "Max angular distance (degrees) for target acquisition",
        FCVAR_NONE, true, 1.0, true, 180.0);
    g_cvTargetZOffset = CreateConVar("sm_nativeaim_target_z_offset", "-12.0",
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
    g_cvDebug = CreateConVar("sm_nativeaim_debug", "0",
        "Print extra debug information",
        FCVAR_NONE, true, 0.0, true, 1.0);

    RegConsoleCmd("sm_nativeaim_active", Cmd_SetActive,
        "Enable/disable native aim assist for yourself (0/1)");
    RegConsoleCmd("sm_nativeaim_mode", Cmd_SetMode,
        "Set native aim mode (raw/smooth/humanised)");
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
    else if (StrEqual(arg, "humanised", false) || StrEqual(arg, "humanized", false))
    {
        g_iMode[client] = view_as<int>(AimMode_Humanised);
    }
    else
    {
        ReplyToCommand(client, "[NativeAim] Unknown mode '%s' (use raw/smooth/humanised)", arg);
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
    if (mode == view_as<int>(AimMode_Raw))       strcopy(ModeName_buf, sizeof(ModeName_buf), "raw");
    else if (mode == view_as<int>(AimMode_Smooth))    strcopy(ModeName_buf, sizeof(ModeName_buf), "smooth");
    else if (mode == view_as<int>(AimMode_Humanised)) strcopy(ModeName_buf, sizeof(ModeName_buf), "humanised");
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
        g_flReactionReadyTime[client] = now + (g_cvReactionMs.FloatValue / 1000.0);
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
    else // Humanised
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

    angles[0] = desiredPitch;
    angles[1] = desiredYaw;
    angles[2] = 0.0;

    float applyAngles[3];
    applyAngles[0] = desiredPitch;
    applyAngles[1] = desiredYaw;
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
