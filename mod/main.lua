local IsaacRL = RegisterMod("Isaac RL Bridge", 1)
local socket = require("socket")

-- Configuration
local HOST = "127.0.0.1"
local PORT = 5555
local FRAME_SKIP = 4  -- Agent makes a decision every 4 game ticks (15 Hz)

-- State variables
local tcp = nil
local connected = false
local waiting_for_restart = false
local current_action = { move_x = 0.0, move_y = 0.0, shoot_x = 0.0, shoot_y = 0.0 }
local tick_counter = 0

-- Convert an integer seed into Isaac's 8-character alphanumeric seed format
local SEED_CHARS = "ABCDEFGHJKLMNPQRSTWXYZ012346789"
local function format_isaac_seed(num)
    local n = math.abs(math.floor(tonumber(num) or 0))
    local chars = {}
    for i = 1, 8 do
        local idx = (n % #SEED_CHARS) + 1
        chars[i] = string.sub(SEED_CHARS, idx, idx)
        n = math.floor(n / #SEED_CHARS)
    end
    return table.concat(chars, "", 1, 4) .. " " .. table.concat(chars, "", 5, 8)
end

-- 1. Connect to Python Server
local function connect_to_python()
    if tcp then tcp:close() end
    tcp = socket.tcp()
    tcp:settimeout(10.0)
    local ok, err = tcp:connect(HOST, PORT)
    if ok then
        tcp:setoption("tcp-nodelay", true)
        connected = true
        waiting_for_restart = false
        Isaac.ConsoleOutput("[IsaacRL] Connected to Python server!\n")
    else
        connected = false
    end
end

-- 2. Serialize Game State
-- Format:
-- tick|hp|max_hp|px|py|vx|vy|tl_x|tl_y|br_x|br_y|is_clear|room_type|enemies|projectiles|doors|pickups
local function serialize_state()
    local game = Game()
    local room = game:GetRoom()
    local player = Isaac.GetPlayer(0)

    local pos = player.Position
    local vel = player.Velocity
    local hp = player:GetHearts() + player:GetSoulHearts()
    local max_hp = player:GetMaxHearts()
    local is_clear = room:IsClear() and 1 or 0
    local room_type = room:GetType()

    local tl = room:GetTopLeftPos()
    local br = room:GetBottomRightPos()

    -- Doors (Slots 0 to 7)
    local doors = {}
    for slot = 0, 7 do
        local door = room:GetDoor(slot)
        if door ~= nil then
            local is_open = door:IsOpen() and 1 or 0
            local target_type = door.TargetRoomType
            table.insert(doors, string.format(
                "%.1f,%.1f,%d,%d",
                door.Position.X, door.Position.Y, is_open, target_type
            ))
        end
    end

    -- Enemies, Projectiles, and Pickups/Items
    local enemies = {}
    local projectiles = {}
    local pickups = {}

    for _, entity in ipairs(Isaac.GetRoomEntities()) do
        if entity:IsActiveEnemy(false) and entity:IsVulnerableEnemy() then
            table.insert(enemies, string.format(
                "%.1f,%.1f,%.2f,%.2f,%.1f",
                entity.Position.X, entity.Position.Y, entity.Velocity.X, entity.Velocity.Y, entity.HitPoints
            ))
        elseif entity.Type == EntityType.ENTITY_PROJECTILE then
            table.insert(projectiles, string.format(
                "%.1f,%.1f,%.2f,%.2f",
                entity.Position.X, entity.Position.Y, entity.Velocity.X, entity.Velocity.Y
            ))
        elseif entity.Type == EntityType.ENTITY_PICKUP then
            -- Filter out empty pedestals (SubType == 0)
            if entity.Variant ~= PickupVariant.PICKUP_COLLECTIBLE or entity.SubType > 0 then
                table.insert(pickups, string.format(
                    "%.1f,%.1f,%d,%d",
                    entity.Position.X, entity.Position.Y, entity.Variant, entity.SubType
                ))
            end
        end
    end

    return string.format(
        "%d|%d|%d|%.1f|%.1f|%.2f|%.2f|%.1f|%.1f|%.1f|%.1f|%d|%d|%s|%s|%s|%s\n",
        tick_counter, hp, max_hp, pos.X, pos.Y, vel.X, vel.Y,
        tl.X, tl.Y, br.X, br.Y, is_clear, room_type,
        table.concat(enemies, ";"),
        table.concat(projectiles, ";"),
        table.concat(doors, ";"),
        table.concat(pickups, ";")
    )
end

-- 3. Parse Python's Action Response
local function parse_action(msg)
    local mx, my, sx, sy = string.match(msg, "([^,]+),([^,]+),([^,]+),([^,]+)")
    if mx then
        current_action.move_x = tonumber(mx)
        current_action.move_y = tonumber(my)
        current_action.shoot_x = tonumber(sx)
        current_action.shoot_y = tonumber(sy)
    end
end

-- 4. Post-Game-Started Hook (Unblocks IPC once the new run is fully loaded)
function IsaacRL:on_game_started(is_continued)
    if waiting_for_restart then
        waiting_for_restart = false
        tick_counter = 0
    end
end

-- 5. Main Simulation Tick (Runs 60 times per second)
function IsaacRL:on_update()
    tick_counter = tick_counter + 1

    if waiting_for_restart then
        return
    end

    -- Attempt reconnect every 60 ticks (1 sec) if disconnected
    if not connected then
        if tick_counter % 60 == 0 then connect_to_python() end
        return
    end

    -- Only query Python every FRAME_SKIP ticks
    if tick_counter % FRAME_SKIP ~= 0 then
        return
    end

    local payload = serialize_state()
    local bytes, err = tcp:send(payload)
    if not bytes then
        connected = false
        Isaac.ConsoleOutput("[IsaacRL] Lost connection: " .. tostring(err) .. "\n")
        return
    end

    -- BLOCKING READ: Game pauses here until Python computes the next action
    local reply, recv_err = tcp:receive("*l")
    if not reply then
        connected = false
        return
    end

    if string.sub(reply, 1, 5) == "RESET" then
        waiting_for_restart = true
        current_action = { move_x = 0.0, move_y = 0.0, shoot_x = 0.0, shoot_y = 0.0 }
        Isaac.ExecuteCommand("restart")
    else
        parse_action(reply)
    end
end

-- 6. Input Injection Hook
function IsaacRL:on_input(entity, hook, action)
    if not connected or entity == nil or entity.Type ~= EntityType.ENTITY_PLAYER then
        return nil
    end

    local val = nil
    if action == ButtonAction.ACTION_LEFT then
        val = current_action.move_x < 0 and math.abs(current_action.move_x) or 0.0
    elseif action == ButtonAction.ACTION_RIGHT then
        val = current_action.move_x > 0 and current_action.move_x or 0.0
    elseif action == ButtonAction.ACTION_UP then
        val = current_action.move_y < 0 and math.abs(current_action.move_y) or 0.0
    elseif action == ButtonAction.ACTION_DOWN then
        val = current_action.move_y > 0 and current_action.move_y or 0.0
    elseif action == ButtonAction.ACTION_SHOOTLEFT then
        val = current_action.shoot_x < 0 and math.abs(current_action.shoot_x) or 0.0
    elseif action == ButtonAction.ACTION_SHOOTRIGHT then
        val = current_action.shoot_x > 0 and current_action.shoot_x or 0.0
    elseif action == ButtonAction.ACTION_SHOOTUP then
        val = current_action.shoot_y < 0 and math.abs(current_action.shoot_y) or 0.0
    elseif action == ButtonAction.ACTION_SHOOTDOWN then
        val = current_action.shoot_y > 0 and current_action.shoot_y or 0.0
    end

    if val == nil then
        return nil
    end

    if hook == InputHook.IS_ACTION_PRESSED or hook == InputHook.IS_ACTION_TRIGGERED then
        return val > 0.5
    elseif hook == InputHook.GET_ACTION_VALUE then
        return val
    end
end

IsaacRL:AddCallback(ModCallbacks.MC_POST_GAME_STARTED, IsaacRL.on_game_started)
IsaacRL:AddCallback(ModCallbacks.MC_POST_UPDATE, IsaacRL.on_update)
IsaacRL:AddCallback(ModCallbacks.MC_INPUT_ACTION, IsaacRL.on_input)