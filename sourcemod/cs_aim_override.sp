#include <sourcemod>
#include <sdktools>
#include <socket>

#pragma semicolon 1
#pragma newdecls required

#define PLUGIN_VERSION "1.0.0"
#define OVERRIDE_PORT 27020

public Plugin myinfo = {
    name = "CS Aim Override",
    author = "Tzan",
    description = "Applies Python-generated view angles to target player",
    version = PLUGIN_VERSION,
    url = ""
};

ConVar g_cvActive;
int g_iTargetPlayer = -1;
Handle g_hSocket = INVALID_HANDLE;
Handle g_hClientSocket = INVALID_HANDLE;

float g_fPendingYaw = 0.0;
float g_fPendingPitch = 0.0;
bool g_bHasPending = false;
int g_iPacketCount = 0;
int g_iApplyCount = 0;
int g_iConnectionCount = 0;

public void OnPluginStart()
{
    g_cvActive = CreateConVar("sm_aim_override_active", "0",
        "Enable/disable aim override (0=off, 1=on)", FCVAR_NONE, true, 0.0, true, 1.0);
    RegAdminCmd("sm_override_target", Cmd_SetTarget, ADMFLAG_ROOT,
        "Set override target player by userid");
    RegConsoleCmd("sm_override_me", Cmd_SetSelfTarget,
        "Set yourself as the aim override target");
    RegConsoleCmd("sm_override_active", Cmd_SetActive,
        "Enable/disable aim override (0=off, 1=on)");
    RegConsoleCmd("sm_override_toggle", Cmd_Toggle,
        "Toggle aim override on/off");
    RegConsoleCmd("sm_override_status", Cmd_Status,
        "Show aim override target, active state, and packet counters");

    g_hSocket = SocketCreate(SOCKET_TCP, OnSocketError);
    SocketSetOption(g_hSocket, SocketReuseAddr, 1);
    SocketBind(g_hSocket, "127.0.0.1", OVERRIDE_PORT);
    SocketListen(g_hSocket, OnSocketIncoming);

    PrintToServer("[AimOverride] Listening on TCP 127.0.0.1:%d", OVERRIDE_PORT);
}

public void OnPluginEnd()
{
    if (g_hClientSocket != INVALID_HANDLE)
    {
        CloseHandle(g_hClientSocket);
        g_hClientSocket = INVALID_HANDLE;
    }
    if (g_hSocket != INVALID_HANDLE)
    {
        CloseHandle(g_hSocket);
        g_hSocket = INVALID_HANDLE;
    }
}

public void OnClientPutInServer(int client)
{
    if (!IsFakeClient(client) && g_iTargetPlayer < 1)
    {
        SetTargetPlayer(client, "auto");
    }
}

public void OnClientDisconnect(int client)
{
    if (client == g_iTargetPlayer)
    {
        g_iTargetPlayer = -1;
        SelectFirstHumanTarget();
    }
}

void SelectFirstHumanTarget()
{
    for (int i = 1; i <= MaxClients; i++)
    {
        if (IsClientInGame(i) && !IsFakeClient(i))
        {
            SetTargetPlayer(i, "auto");
            return;
        }
    }
}

void SetTargetPlayer(int target, const char[] source)
{
    g_iTargetPlayer = target;
    PrintToServer("[AimOverride] Target set to player %N (userid %d, source=%s)",
        target, GetClientUserId(target), source);
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
        SetTargetPlayer(target, "command");
        ReplyToCommand(client, "[AimOverride] Target set to player %d (userid %d)", target, userid);
    }
    else
    {
        ReplyToCommand(client, "[AimOverride] Invalid userid %d", userid);
    }
    return Plugin_Handled;
}

public Action Cmd_SetSelfTarget(int client, int args)
{
    if (client <= 0 || !IsClientInGame(client) || IsFakeClient(client))
    {
        ReplyToCommand(client, "[AimOverride] This command must be run by a player.");
        return Plugin_Handled;
    }

    SetTargetPlayer(client, "self");
    ReplyToCommand(client, "[AimOverride] Target set to you (userid %d)", GetClientUserId(client));
    return Plugin_Handled;
}

public Action Cmd_SetActive(int client, int args)
{
    if (args < 1)
    {
        ReplyToCommand(client, "[AimOverride] Active: %d", g_cvActive.BoolValue ? 1 : 0);
        return Plugin_Handled;
    }

    char arg[8];
    GetCmdArg(1, arg, sizeof(arg));
    bool active = StringToInt(arg) != 0;
    g_cvActive.SetBool(active);
    ReplyToCommand(client, "[AimOverride] Active: %d", active ? 1 : 0);
    return Plugin_Handled;
}

public Action Cmd_Toggle(int client, int args)
{
    bool active = !g_cvActive.BoolValue;
    g_cvActive.SetBool(active);
    ReplyToCommand(client, "[AimOverride] Active: %d", active ? 1 : 0);
    return Plugin_Handled;
}

public Action Cmd_Status(int client, int args)
{
    int userid = 0;
    if (g_iTargetPlayer > 0 && IsClientInGame(g_iTargetPlayer))
    {
        userid = GetClientUserId(g_iTargetPlayer);
    }

    ReplyToCommand(client,
        "[AimOverride] active=%d target_slot=%d target_userid=%d connected=%d pending=%d packets=%d applied=%d last_yaw=%.2f last_pitch=%.2f",
        g_cvActive.BoolValue ? 1 : 0,
        g_iTargetPlayer,
        userid,
        (g_hClientSocket != INVALID_HANDLE && SocketIsConnected(g_hClientSocket)) ? 1 : 0,
        g_bHasPending ? 1 : 0,
        g_iPacketCount,
        g_iApplyCount,
        g_fPendingYaw,
        g_fPendingPitch);
    return Plugin_Handled;
}

void NormaliseAimAngles(float &yaw, float &pitch)
{
    while (yaw > 180.0)
    {
        yaw -= 360.0;
    }
    while (yaw < -180.0)
    {
        yaw += 360.0;
    }

    if (pitch > 89.0)
    {
        pitch = 89.0;
    }
    else if (pitch < -89.0)
    {
        pitch = -89.0;
    }
}

void ParseAimPayload(const char[] data, int dataSize)
{
    char payload[128];
    int copySize = dataSize;
    if (copySize >= sizeof(payload))
    {
        copySize = sizeof(payload) - 1;
    }

    for (int i = 0; i < copySize; i++)
    {
        payload[i] = data[i];
    }
    payload[copySize] = '\0';
    TrimString(payload);

    int lineStart = 0;
    int payloadLen = strlen(payload);
    for (int i = 0; i <= payloadLen; i++)
    {
        if (payload[i] != '\n' && payload[i] != '\0')
        {
            continue;
        }

        payload[i] = '\0';
        char line[64];
        strcopy(line, sizeof(line), payload[lineStart]);
        TrimString(line);
        lineStart = i + 1;

        if (line[0] == '\0')
        {
            continue;
        }

        char parts[2][32];
        if (ExplodeString(line, " ", parts, sizeof(parts), sizeof(parts[])) < 2)
        {
            PrintToServer("[AimOverride] Ignored invalid aim payload: %s", line);
            continue;
        }

        g_fPendingYaw = StringToFloat(parts[0]);
        g_fPendingPitch = StringToFloat(parts[1]);
        NormaliseAimAngles(g_fPendingYaw, g_fPendingPitch);
        g_bHasPending = true;
        g_iPacketCount++;

        if (g_iPacketCount <= 5 || g_iPacketCount % 100 == 0)
        {
            PrintToServer("[AimOverride] Aim packet #%d yaw=%.2f pitch=%.2f",
                g_iPacketCount, g_fPendingYaw, g_fPendingPitch);
        }
    }
}

public void OnSocketIncoming(Handle socket, Handle newSocket, const char[] remoteIP, int remotePort, int arg)
{
    if (g_hClientSocket != INVALID_HANDLE)
    {
        CloseHandle(g_hClientSocket);
    }

    g_hClientSocket = newSocket;
    g_iConnectionCount++;
    SocketSetReceiveCallback(g_hClientSocket, OnSocketReceive);
    SocketSetDisconnectCallback(g_hClientSocket, OnSocketDisconnect);
    SocketSetErrorCallback(g_hClientSocket, OnSocketError);

    PrintToServer("[AimOverride] Python override connected #%d from %s:%d",
        g_iConnectionCount, remoteIP, remotePort);
}

public void OnSocketReceive(Handle socket, const char[] data, int dataSize, int arg)
{
    if (dataSize <= 0)
    {
        return;
    }

    ParseAimPayload(data, dataSize);
}

public void OnSocketDisconnect(Handle socket, int arg)
{
    if (socket == g_hClientSocket)
    {
        g_hClientSocket = INVALID_HANDLE;
        PrintToServer("[AimOverride] Python override disconnected");
    }
}

public void OnSocketError(Handle socket, int errorType, int errorNum, int arg)
{
    PrintToServer("[AimOverride] Socket error: type=%d num=%d", errorType, errorNum);
    if (socket == g_hClientSocket)
    {
        g_hClientSocket = INVALID_HANDLE;
    }
}

public Action OnPlayerRunCmd(int client, int &buttons, int &impulse,
    float vel[3], float angles[3], int &weapon,
    int &subtype, int &cmdnum, int &tickcount, int &seed, int mouse[2])
{
    if (client != g_iTargetPlayer) return Plugin_Continue;
    if (!g_cvActive.BoolValue) return Plugin_Continue;
    if (!g_bHasPending) return Plugin_Continue;

    float newAngles[3];
    newAngles[0] = g_fPendingPitch;
    newAngles[1] = g_fPendingYaw;
    newAngles[2] = 0.0;

    angles[0] = newAngles[0];
    angles[1] = newAngles[1];
    angles[2] = newAngles[2];
    TeleportEntity(client, NULL_VECTOR, newAngles, NULL_VECTOR);

    g_bHasPending = false;
    g_iApplyCount++;

    if (g_iApplyCount <= 5 || g_iApplyCount % 100 == 0)
    {
        PrintToServer("[AimOverride] Applied #%d to %N yaw=%.2f pitch=%.2f",
            g_iApplyCount, client, newAngles[1], newAngles[0]);
    }

    return Plugin_Changed;
}
