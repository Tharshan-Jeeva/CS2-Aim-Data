#include <sourcemod>
#include <sdktools>
#include <cstrike>

#pragma semicolon 1
#pragma newdecls required

#define PLUGIN_VERSION "1.0.0"

// Delay before a dead bot is respawned. Short enough that there is always a
// target on screen, long enough that the death is visible.
#define RESPAWN_DELAY     1.5
// Delay before the round/map resets after the VISITOR dies, so the kill is
// visible before everything snaps back.
#define RESET_DELAY       1.5
// How often the on-screen "current aim style" hint is refreshed. Hint text
// fades on its own, so we re-print it on a gentle cadence to keep it visible.
#define HINT_INTERVAL     5.0
// How often the chat reminder of the controls is printed.
#define REMINDER_INTERVAL 30.0

public Plugin myinfo = {
    name = "CS Aim Kiosk",
    author = "Tharshan-Jeeva",
    description = "Exhibition kiosk: AK-only spawns, instant respawn, on-screen aim-style cycle (drives cs_aim_live_controller)",
    version = PLUGIN_VERSION,
    url = ""
};

// Aim styles, in cycle order. The command token is what we feed to the
// SM-native controller's `sm_nativeaim_mode`; the name is what the visitor
// sees on screen. THESE MUST MATCH cs_aim_live_controller.sp's Cmd_SetMode.
char g_sModeCmd[4][16]  = { "raw",          "smooth", "humanised",  "humanised_high"   };
char g_sModeName[4][20] = { "RAW (robotic)", "SMOOTH", "HUMANISED",  "HUMANISED-HIGH"   };

// Kiosk-owned per-client style index. The controller owns the real mode; we
// keep a parallel index purely to know what to show and what to cycle to.
int g_iKioskMode[MAXPLAYERS + 1];

ConVar g_cvEnable;
ConVar g_cvTeam;            // team to lock visitors onto (2 = T, 3 = CT). Bots are CT.
ConVar g_cvRoundsPerReset; // reload the map after this many visitor deaths (0 = never)

// Counts visitor deaths (= round resets) since the last full map reload.
int g_iRoundsSinceReset = 0;

public void OnPluginStart()
{
    g_cvEnable = CreateConVar("sm_kiosk_enable", "1",
        "Master enable for exhibition kiosk behaviour (0=off, 1=on)",
        FCVAR_NONE, true, 0.0, true, 1.0);
    g_cvTeam = CreateConVar("sm_kiosk_team", "2",
        "Team to place human visitors on (2=T, 3=CT). Bots take the other side.",
        FCVAR_NONE, true, 1.0, true, 3.0);
    g_cvRoundsPerReset = CreateConVar("sm_kiosk_rounds_per_reset", "30",
        "Reload the map after this many visitor deaths/round-resets to clear accumulated state (0 = never)",
        FCVAR_NONE, true, 0.0, true, 1000.0);

    RegConsoleCmd("sm_kiosk_cycle", Cmd_Cycle,
        "Cycle aim style: raw -> smooth -> humanised -> humanised_high");

    HookEvent("player_spawn", Event_PlayerSpawn, EventHookMode_Post);
    HookEvent("player_death", Event_PlayerDeath, EventHookMode_Post);

    CreateTimer(HINT_INTERVAL, Timer_Hint, _, TIMER_REPEAT);
    CreateTimer(REMINDER_INTERVAL, Timer_Reminder, _, TIMER_REPEAT);

    for (int i = 1; i <= MAXPLAYERS; i++)
    {
        g_iKioskMode[i] = 0;
    }
}

// Runs AFTER all server/map configs have exec'd and SourceMod is fully loaded.
// This is the correct place for SourceMod-dependent setup that cannot live in
// exhibition_server.cfg (that file is exec'd by the engine before SM exists).
public void OnConfigsExecuted()
{
    if (!g_cvEnable.BoolValue) return;

    // The in-engine SM-native controller is the only aimbot used here.
    ServerCommand("sm_nativeaim_enable 1");

    // No telemetry/recording at the exhibition: unload the telemetry plugin so
    // it stops POSTing every tick to a dead 127.0.0.1:3000, and unload the
    // legacy Python-TCP override plugin (never used in kiosk mode). Unloading
    // an already-absent plugin is a harmless no-op warning.
    ServerCommand("sm plugins unload cs_aim_telemetry");
    ServerCommand("sm plugins unload cs_aim_override");
}

public void OnClientPutInServer(int client)
{
    if (!g_cvEnable.BoolValue) return;
    if (client <= 0 || client > MaxClients) return;
    if (IsFakeClient(client)) return;

    g_iKioskMode[client] = 0;
    // Place the visitor on the human side and start them in RAW. A short delay
    // lets the client finish entering the game before we switch team / spawn.
    CreateTimer(1.0, Timer_InitVisitor, GetClientUserId(client));
}

public Action Timer_InitVisitor(Handle timer, int userid)
{
    int client = GetClientOfUserId(userid);
    if (client <= 0 || !IsClientInGame(client) || IsFakeClient(client))
        return Plugin_Stop;

    int team = g_cvTeam.IntValue;
    if (GetClientTeam(client) != team)
    {
        CS_SwitchTeam(client, team);
    }
    if (!IsPlayerAlive(client))
    {
        CS_RespawnPlayer(client);
    }
    // Start the controller in RAW for this client so the first thing they feel
    // is the most obvious (robotic) assist.
    SetControllerMode(client, 0);
    return Plugin_Stop;
}

public void Event_PlayerSpawn(Event event, const char[] name, bool dontBroadcast)
{
    if (!g_cvEnable.BoolValue) return;

    int client = GetClientOfUserId(event.GetInt("userid"));
    if (client <= 0 || !IsClientInGame(client)) return;
    if (IsFakeClient(client)) return; // bots keep their own buy logic

    // Loadout is assigned by the game slightly after the spawn event fires, so
    // strip + give on the next frame to guarantee our weapon sticks.
    CreateTimer(0.1, Timer_EquipVisitor, GetClientUserId(client));
}

public Action Timer_EquipVisitor(Handle timer, int userid)
{
    int client = GetClientOfUserId(userid);
    if (client <= 0 || !IsClientInGame(client) || IsFakeClient(client))
        return Plugin_Stop;
    if (!IsPlayerAlive(client))
        return Plugin_Stop;

    // AK-47 only (matches the study weapon). Strip everything, then give a
    // knife + AK and make the AK active.
    StripWeapons(client);
    GivePlayerItem(client, "weapon_knife");
    int ak = GivePlayerItem(client, "weapon_ak47");
    if (ak != -1)
    {
        // GivePlayerItem gives the AK its full default reserve (90), so the
        // visitor never runs dry over a long session. Make the AK active.
        EquipPlayerWeapon(client, ak);
    }
    return Plugin_Stop;
}

void StripWeapons(int client)
{
    for (int slot = 0; slot <= 5; slot++)
    {
        int wep = GetPlayerWeaponSlot(client, slot);
        while (wep != -1)
        {
            RemovePlayerItem(client, wep);
            RemoveEdict(wep);
            wep = GetPlayerWeaponSlot(client, slot);
        }
    }
}

public void Event_PlayerDeath(Event event, const char[] name, bool dontBroadcast)
{
    if (!g_cvEnable.BoolValue) return;

    int client = GetClientOfUserId(event.GetInt("userid"));
    if (client <= 0 || !IsClientInGame(client)) return;

    if (IsFakeClient(client))
    {
        // A bot died: CS:S bots don't auto-respawn, so put it back quickly to
        // keep a target on screen while the visitor is alive.
        CreateTimer(RESPAWN_DELAY, Timer_RespawnBot, GetClientUserId(client));
        return;
    }

    // The VISITOR died: reset the round so everything snaps back to a clean
    // start. Every Nth death (sm_kiosk_rounds_per_reset) we instead reload the
    // map entirely, to clear any accumulated server state on a long unattended
    // run (the client stays connected through a changelevel).
    g_iRoundsSinceReset++;
    CreateTimer(RESET_DELAY, Timer_VisitorReset);
}

public Action Timer_RespawnBot(Handle timer, int userid)
{
    int client = GetClientOfUserId(userid);
    if (client <= 0 || !IsClientInGame(client) || !IsFakeClient(client))
        return Plugin_Stop;
    if (IsPlayerAlive(client))
        return Plugin_Stop;

    int team = GetClientTeam(client);
    if (team != CS_TEAM_T && team != CS_TEAM_CT)
        return Plugin_Stop;

    CS_RespawnPlayer(client);
    return Plugin_Stop;
}

public Action Timer_VisitorReset(Handle timer)
{
    int per = g_cvRoundsPerReset.IntValue;
    if (per > 0 && g_iRoundsSinceReset >= per)
    {
        // Periodic full reset: reload the current map. Recreates all entities
        // and resets all game state without disconnecting the client.
        g_iRoundsSinceReset = 0;
        char map[64];
        GetCurrentMap(map, sizeof(map));
        PrintToChatAll("\x04[AIM STUDY]\x01 Periodic reset — reloading the map...");
        ServerCommand("changelevel %s", map);
    }
    else
    {
        // Normal case: just restart the round (respawns the visitor + bots at
        // spawn points; mp_freezetime is 0 so play resumes immediately).
        ServerCommand("mp_restartgame 1");
    }
    return Plugin_Stop;
}

public Action Cmd_Cycle(int client, int args)
{
    if (!g_cvEnable.BoolValue)
        return Plugin_Handled;
    if (client <= 0 || !IsClientInGame(client) || IsFakeClient(client))
        return Plugin_Handled;

    g_iKioskMode[client] = (g_iKioskMode[client] + 1) % 4;
    int idx = g_iKioskMode[client];
    SetControllerMode(client, idx);

    PrintHintText(client, "AIM STYLE: %s   (%d/4)\nHOLD V = assist  |  F = change style",
        g_sModeName[idx], idx + 1);
    // Audible feedback so the visitor knows the press registered.
    ClientCommand(client, "play buttons/button14.wav");
    return Plugin_Handled;
}

// Drive the SM-native aim controller (cs_aim_live_controller.sp) by running its
// console command in the client's own context. We never touch the Python TCP
// bot path — the in-engine controller is the only aimbot used here.
void SetControllerMode(int client, int idx)
{
    char cmd[48];
    Format(cmd, sizeof(cmd), "sm_nativeaim_mode %s", g_sModeCmd[idx]);
    FakeClientCommand(client, cmd);
}

public Action Timer_Hint(Handle timer)
{
    if (!g_cvEnable.BoolValue)
        return Plugin_Continue;

    for (int i = 1; i <= MaxClients; i++)
    {
        if (!IsClientInGame(i) || IsFakeClient(i) || !IsPlayerAlive(i))
            continue;
        int idx = g_iKioskMode[i];
        PrintHintText(i, "AIM STYLE: %s   (%d/4)\nHOLD V = assist  |  F = change style",
            g_sModeName[idx], idx + 1);
    }
    return Plugin_Continue;
}

public Action Timer_Reminder(Handle timer)
{
    if (!g_cvEnable.BoolValue)
        return Plugin_Continue;

    for (int i = 1; i <= MaxClients; i++)
    {
        if (!IsClientInGame(i) || IsFakeClient(i))
            continue;
        PrintToChat(i, "\x04[AIM STUDY]\x01 HOLD \x04V\x01 = aim assist ON (release = you, the human).");
        PrintToChat(i, "\x04[AIM STUDY]\x01 Press \x04F\x01 to change style: raw \xE2\x86\x92 smooth \xE2\x86\x92 humanised \xE2\x86\x92 high.");
    }
    return Plugin_Continue;
}
