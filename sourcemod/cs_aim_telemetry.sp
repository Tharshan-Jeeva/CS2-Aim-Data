#include <sourcemod>
#include <sdktools>
#include <SteamWorks>

#pragma semicolon 1
#pragma newdecls required

#define PLUGIN_VERSION "1.0.0"
#define TELEMETRY_URL "http://127.0.0.1:3000/event"
#define HEARTBEAT_INTERVAL 0.5

public Plugin myinfo = {
    name = "CS Aim Telemetry",
    author = "Tzan",
    description = "Streams per-tick view angles, kills, fires, hits to Python telemetry server",
    version = PLUGIN_VERSION,
    url = ""
};

int g_iTargetPlayer = -1;
Handle g_hHeartbeatTimer = INVALID_HANDLE;

public void OnPluginStart()
{
    RegAdminCmd("sm_telemetry_target", Cmd_SetTarget, ADMFLAG_ROOT, "Set the player to track");
    RegConsoleCmd("sm_telemetry_me", Cmd_SetSelfTarget, "Track yourself for aim telemetry");

    HookEvent("player_death", Event_PlayerDeath);
    HookEvent("weapon_fire", Event_WeaponFire);
    HookEvent("player_hurt", Event_PlayerHurt);
    HookEvent("round_start", Event_RoundStart);
    HookEvent("round_end", Event_RoundEnd);

    g_hHeartbeatTimer = CreateTimer(HEARTBEAT_INTERVAL, Timer_Heartbeat, _, TIMER_REPEAT);
}

public void OnPluginEnd()
{
    if (g_hHeartbeatTimer != INVALID_HANDLE)
    {
        KillTimer(g_hHeartbeatTimer);
        g_hHeartbeatTimer = INVALID_HANDLE;
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
    PrintToServer("[Telemetry] Tracking player %N (userid %d, source=%s)",
        target, GetClientUserId(target), source);
}

public Action Cmd_SetTarget(int client, int args)
{
    if (args < 1)
    {
        ReplyToCommand(client, "[Telemetry] Usage: sm_telemetry_target <userid>");
        return Plugin_Handled;
    }

    char arg[8];
    GetCmdArg(1, arg, sizeof(arg));
    int userid = StringToInt(arg);
    int target = GetClientOfUserId(userid);

    if (target > 0 && IsClientInGame(target))
    {
        SetTargetPlayer(target, "command");
        ReplyToCommand(client, "[Telemetry] Now tracking player %d (userid %d)", target, userid);
    }
    else
    {
        ReplyToCommand(client, "[Telemetry] Invalid userid %d", userid);
    }
    return Plugin_Handled;
}

public Action Cmd_SetSelfTarget(int client, int args)
{
    if (client <= 0 || !IsClientInGame(client) || IsFakeClient(client))
    {
        ReplyToCommand(client, "[Telemetry] This command must be run by a player.");
        return Plugin_Handled;
    }

    SetTargetPlayer(client, "self");
    ReplyToCommand(client, "[Telemetry] Now tracking you (userid %d)", GetClientUserId(client));
    return Plugin_Handled;
}

void SendJSON(const char[] json)
{
    Handle hRequest = SteamWorks_CreateHTTPRequest(k_EHTTPMethodPOST, TELEMETRY_URL);
    if (hRequest == INVALID_HANDLE) return;

    SteamWorks_SetHTTPRequestRawPostBody(hRequest, "application/json", json, strlen(json));
    SteamWorks_SetHTTPRequestNetworkActivityTimeout(hRequest, 2);
    SteamWorks_SendHTTPRequest(hRequest);
    delete hRequest;
}

public Action OnPlayerRunCmd(int client, int &buttons, int &impulse,
    float vel[3], float angles[3], int &weapon,
    int &subtype, int &cmdnum, int &tickcount, int &seed, int mouse[2])
{
    if (client != g_iTargetPlayer || !IsPlayerAlive(client))
        return Plugin_Continue;

    float pos[3], eyePos[3], velocity[3], eyeAngles[3];
    GetClientAbsOrigin(client, pos);
    GetClientEyePosition(client, eyePos);
    GetEntPropVector(client, Prop_Data, "m_vecVelocity", velocity);
    GetClientEyeAngles(client, eyeAngles);

    char enemies[3072];
    enemies[0] = '\0';
    int enemyCount = 0;

    for (int i = 1; i <= MaxClients; i++)
    {
        if (i == client || !IsClientInGame(i) || !IsPlayerAlive(i))
            continue;
        if (GetClientTeam(i) == GetClientTeam(client))
            continue;

        float enemyOrigin[3];
        GetClientAbsOrigin(i, enemyOrigin);

        // Compute synthetic head-centre aim point.
        // Origin XY is the true model centre (no facing-direction bias).
        float enemyAimPoint[3];
        enemyAimPoint[0] = enemyOrigin[0];
        enemyAimPoint[1] = enemyOrigin[1];
        bool isDucking = (GetEntityFlags(i) & FL_DUCKING) != 0;
        enemyAimPoint[2] = enemyOrigin[2] + (isDucking ? 48.0 : 65.0);

        float enemyVelocity[3];
        GetEntPropVector(i, Prop_Data, "m_vecVelocity", enemyVelocity);

        bool visible = CanSeeTarget(client, enemyAimPoint);
        int health = GetClientHealth(i);

        // origin = server feet origin; aim_position = computed head centre;
        // position = aim_position kept for backward compatibility.
        char entry[384];
        Format(entry, sizeof(entry),
            "%s{\"id\":%d,\"origin\":[%.1f,%.1f,%.1f],\"aim_position\":[%.1f,%.1f,%.1f],\"position\":[%.1f,%.1f,%.1f],\"velocity\":[%.1f,%.1f,%.1f],\"visible\":%s,\"health\":%d}",
            enemyCount > 0 ? "," : "",
            i,
            enemyOrigin[0], enemyOrigin[1], enemyOrigin[2],
            enemyAimPoint[0], enemyAimPoint[1], enemyAimPoint[2],
            enemyAimPoint[0], enemyAimPoint[1], enemyAimPoint[2],
            enemyVelocity[0], enemyVelocity[1], enemyVelocity[2],
            visible ? "true" : "false", health);
        StrCat(enemies, sizeof(enemies), entry);
        enemyCount++;
    }

    char json[5120];
    Format(json, sizeof(json),
        "{\"type\":\"tick\",\"tick\":%d,\"timestamp_server\":%.3f,"
    ... "\"player_id\":%d,\"position\":[%.1f,%.1f,%.1f],"
    ... "\"eye_position\":[%.1f,%.1f,%.1f],"
    ... "\"velocity\":[%.1f,%.1f,%.1f],\"view_angles\":[%.2f,%.2f],"
    ... "\"buttons\":{\"fire\":%d,\"jump\":%d,\"duck\":%d,\"walk\":%d,"
    ... "\"forward\":%d,\"back\":%d,\"left\":%d,\"right\":%d},"
    ... "\"enemies\":[%s]}",
        GetGameTickCount(), GetGameTime(),
        client, pos[0], pos[1], pos[2],
        eyePos[0], eyePos[1], eyePos[2],
        velocity[0], velocity[1], velocity[2],
        eyeAngles[0], eyeAngles[1],
        (buttons & IN_ATTACK) ? 1 : 0,
        (buttons & IN_JUMP) ? 1 : 0,
        (buttons & IN_DUCK) ? 1 : 0,
        (buttons & IN_SPEED) ? 1 : 0,
        (buttons & IN_FORWARD) ? 1 : 0,
        (buttons & IN_BACK) ? 1 : 0,
        (buttons & IN_MOVELEFT) ? 1 : 0,
        (buttons & IN_MOVERIGHT) ? 1 : 0,
        enemies);

    SendJSON(json);
    return Plugin_Continue;
}

bool CanSeeTarget(int client, float targetPos[3])
{
    float eyePos[3];
    GetClientEyePosition(client, eyePos);

    Handle trace = TR_TraceRayFilterEx(eyePos, targetPos, MASK_VISIBLE, RayType_EndPoint, TraceFilter_NoPlayers);
    bool hit = TR_DidHit(trace);
    delete trace;

    return !hit;
}

public bool TraceFilter_NoPlayers(int entity, int contentsMask)
{
    return entity > MaxClients;
}

public void Event_PlayerDeath(Event event, const char[] name, bool dontBroadcast)
{
    int attacker = GetClientOfUserId(event.GetInt("attacker"));
    if (attacker != g_iTargetPlayer) return;

    int victim = GetClientOfUserId(event.GetInt("userid"));
    float attackerPos[3], victimPos[3], attackerAngles[3];
    GetClientAbsOrigin(attacker, attackerPos);
    GetClientAbsOrigin(victim, victimPos);
    GetClientEyeAngles(attacker, attackerAngles);

    char weapon[32];
    event.GetString("weapon", weapon, sizeof(weapon));
    bool headshot = event.GetBool("headshot");

    char json[512];
    Format(json, sizeof(json),
        "{\"type\":\"kill\",\"tick\":%d,\"timestamp_server\":%.3f,"
    ... "\"attacker_id\":%d,\"victim_id\":%d,"
    ... "\"attacker_position\":[%.1f,%.1f,%.1f],"
    ... "\"victim_position\":[%.1f,%.1f,%.1f],"
    ... "\"attacker_angles\":[%.2f,%.2f],"
    ... "\"weapon\":\"%s\",\"headshot\":%d}",
        GetGameTickCount(), GetGameTime(),
        attacker, victim,
        attackerPos[0], attackerPos[1], attackerPos[2],
        victimPos[0], victimPos[1], victimPos[2],
        attackerAngles[0], attackerAngles[1],
        weapon, headshot ? 1 : 0);

    SendJSON(json);
}

public void Event_WeaponFire(Event event, const char[] name, bool dontBroadcast)
{
    int shooter = GetClientOfUserId(event.GetInt("userid"));
    if (shooter != g_iTargetPlayer) return;

    float pos[3], angles[3];
    GetClientAbsOrigin(shooter, pos);
    GetClientEyeAngles(shooter, angles);

    char weapon[32];
    event.GetString("weapon", weapon, sizeof(weapon));

    char json[256];
    Format(json, sizeof(json),
        "{\"type\":\"weapon_fire\",\"tick\":%d,\"timestamp_server\":%.3f,"
    ... "\"shooter_id\":%d,\"weapon\":\"%s\","
    ... "\"position\":[%.1f,%.1f,%.1f],\"view_angles\":[%.2f,%.2f]}",
        GetGameTickCount(), GetGameTime(),
        shooter, weapon,
        pos[0], pos[1], pos[2],
        angles[0], angles[1]);

    SendJSON(json);
}

public void Event_PlayerHurt(Event event, const char[] name, bool dontBroadcast)
{
    int attacker = GetClientOfUserId(event.GetInt("attacker"));
    if (attacker != g_iTargetPlayer) return;

    int victim = GetClientOfUserId(event.GetInt("userid"));
    int damage = event.GetInt("dmg_health");
    int hitgroup = event.GetInt("hitgroup");

    char weapon[32];
    event.GetString("weapon", weapon, sizeof(weapon));

    char json[256];
    Format(json, sizeof(json),
        "{\"type\":\"player_hurt\",\"tick\":%d,\"timestamp_server\":%.3f,"
    ... "\"attacker_id\":%d,\"victim_id\":%d,"
    ... "\"damage\":%d,\"hitgroup\":%d,\"weapon\":\"%s\"}",
        GetGameTickCount(), GetGameTime(),
        attacker, victim, damage, hitgroup, weapon);

    SendJSON(json);
}

public void Event_RoundStart(Event event, const char[] name, bool dontBroadcast)
{
    char json[128];
    Format(json, sizeof(json),
        "{\"type\":\"round_start\",\"tick\":%d,\"timestamp_server\":%.3f}",
        GetGameTickCount(), GetGameTime());
    SendJSON(json);
}

public void Event_RoundEnd(Event event, const char[] name, bool dontBroadcast)
{
    int winner = event.GetInt("winner");
    char json[128];
    Format(json, sizeof(json),
        "{\"type\":\"round_end\",\"tick\":%d,\"timestamp_server\":%.3f,\"winner\":%d}",
        GetGameTickCount(), GetGameTime(), winner);
    SendJSON(json);
}

public Action Timer_Heartbeat(Handle timer)
{
    if (g_iTargetPlayer < 1 || !IsClientInGame(g_iTargetPlayer))
        return Plugin_Continue;

    int health = GetClientHealth(g_iTargetPlayer);
    int armor = GetClientArmor(g_iTargetPlayer);
    int kills = GetClientFrags(g_iTargetPlayer);
    int deaths = GetClientDeaths(g_iTargetPlayer);

    char json[256];
    Format(json, sizeof(json),
        "{\"type\":\"heartbeat\",\"tick\":%d,\"timestamp_server\":%.3f,"
    ... "\"health\":%d,\"armor\":%d,\"kills\":%d,\"deaths\":%d}",
        GetGameTickCount(), GetGameTime(),
        health, armor, kills, deaths);

    SendJSON(json);
    return Plugin_Continue;
}
