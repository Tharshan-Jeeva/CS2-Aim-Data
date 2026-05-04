#include <sourcemod>
#include <sdktools>
#include <socket>

#pragma semicolon 1
#pragma newdecls required

#define PLUGIN_VERSION "1.0.0"
#define UDP_PORT 27020

public Plugin myinfo = {
    name = "CS Aim Override",
    author = "Tzan",
    description = "Applies Python-generated view angles to target player via UDP",
    version = PLUGIN_VERSION,
    url = ""
};

ConVar g_cvActive;
int g_iTargetPlayer = -1;
Handle g_hSocket = INVALID_HANDLE;

float g_fPendingYaw = 0.0;
float g_fPendingPitch = 0.0;
bool g_bHasPending = false;

public void OnPluginStart()
{
    g_cvActive = CreateConVar("sm_aim_override_active", "0",
        "Enable/disable aim override (0=off, 1=on)", FCVAR_NONE, true, 0.0, true, 1.0);
    RegAdminCmd("sm_override_target", Cmd_SetTarget, ADMFLAG_ROOT,
        "Set override target player by userid");

    g_hSocket = SocketCreate(SOCKET_UDP, OnSocketError);
    SocketBind(g_hSocket, "127.0.0.1", UDP_PORT);
    SocketSetReceiveCallback(g_hSocket, OnSocketReceive);

    PrintToServer("[AimOverride] Listening on UDP 127.0.0.1:%d", UDP_PORT);
}

public Action Cmd_SetTarget(int client, int args)
{
    if (args < 1)
    {
        ReplyToCommand(client, "[AimOverride] Usage: sm_override_target <userid>");
        return Plugin_Handled;
    }
    char arg[8];
    GetCmdArg(1, arg, sizeof(arg));
    int userid = StringToInt(arg);
    int target = GetClientOfUserId(userid);
    if (target > 0 && IsClientInGame(target))
    {
        g_iTargetPlayer = target;
        ReplyToCommand(client, "[AimOverride] Target set to player %d (userid %d)", target, userid);
    }
    else
    {
        ReplyToCommand(client, "[AimOverride] Invalid userid %d", userid);
    }
    return Plugin_Handled;
}

public void OnSocketReceive(Handle socket, const char[] data, int dataSize, int arg)
{
    if (dataSize >= 8)
    {
        g_fPendingYaw = view_as<float>(
            data[0] | (data[1] << 8) | (data[2] << 16) | (data[3] << 24));
        g_fPendingPitch = view_as<float>(
            data[4] | (data[5] << 8) | (data[6] << 16) | (data[7] << 24));
        g_bHasPending = true;
    }
}

public void OnSocketError(Handle socket, int errorType, int errorNum, int arg)
{
    PrintToServer("[AimOverride] Socket error: type=%d num=%d", errorType, errorNum);
}

public Action OnPlayerRunCmd(int client, int &buttons, int &impulse,
    float vel[3], float angles[3], int &weapon,
    int &subtype, int &cmdnum, int &tickcount, int &seed, int mouse[2])
{
    if (client != g_iTargetPlayer) return Plugin_Continue;
    if (!g_cvActive.BoolValue) return Plugin_Continue;
    if (!g_bHasPending) return Plugin_Continue;

    angles[0] = g_fPendingPitch;
    angles[1] = g_fPendingYaw;
    g_bHasPending = false;

    return Plugin_Changed;
}
