local IsaacRL = RegisterMod("Isaac RL Bridge", 1)
local socket = require("socket")

-- Configuration
local HOST = "127.0.0.1"
local PORT = 5555
local FRAME_SKIP = 4  -- Agent makes a decision every 4 game ticks (15 Hz)

-- State variables
local tcp = nil
local connected = false
local current_action = { move_x = 0.0, move_y = 0.0, shoot_x = 0.0, shoot_y = 0.0 }
local tick_counter = 0

-- 1. Connect to Python Server
local function connect_to_python()
    if tcp then tcp:close() end
    tcp = socket.tcp()
    tcp:settimeout(2.0) -- 2 second timeout so the game doesn't freeze permanently if Python stops
    local ok, err = tcp:connect(HOST, PORT)
    if ok then
        tcp:setoption("tcp-nodelay", true) -- Disable Nagle's algorithm for minimal IPC latency
        connected = true
        Isaac.ConsoleOutput("[IsaacRL] Connected to Python server!\n")
    else
        connected = false
    end
end

-- 2. Serialize Game State into a compact flat string (fast & zero external JSON dependencies)
-- Format: "tick|hp|max_hp|px|py|vx|vy|room_clear|enemy_count|ex1,ey1,ehp1;ex2,ey2,ehp2..."
local function serialize_state()
    local game = Game()
    local room = game:GetRoom()
    local player = Isaac.GetPlayer(0)
    
    local pos = player.Position
    local vel = player.Velocity
    local hp = player:GetHearts() + player:GetSoulHearts()
    local max_hp = player:GetMaxHearts()
    local is_clear = room:IsClear() and 1 or 0
    
    local enemies = {}
    for _, entity in ipairs(Isaac.GetRoomEntities()) do
        -- Filter for active, vulnerable enemies (exclude tears, pickups, and decorative props)
        if entity:IsActiveEnemy(false) and entity:IsVulnerableEnemy() then
            table.insert(enemies, string.format("%.1f,%.1f,%.1f", entity.Position.X, entity.Position.Y, entity.HitPoints))
        end
    end
    
    local enemy_str = table.concat(enemies, ";")
    return string.format(
        "%d|%d|%d|%.1f|%.1f|%.2f|%.2f|%d|%d|%s\n",
        tick_counter, hp, max_hp, pos.X, pos.Y, vel.X, vel.Y, is_clear, #enemies, enemy_str
    )
end

-- 3. Parse Python's Action Response
-- Expected format from Python: "move_x,move_y,shoot_x,shoot_y" (e.g., "-1.0,0.0,0.0,1.0")
local function parse_action(msg)
    local mx, my, sx, sy = string.match(msg, "([^,]+),([^,]+),([^,]+),([^,]+)")
    if mx then
        current_action.move_x = tonumber(mx)
        current_action.move_y = tonumber(my)
        current_action.shoot_x = tonumber(sx)
        current_action.shoot_y = tonumber(sy)
    end
end

-- 4. Main Simulation Tick (Runs 60 times per second)
function IsaacRL:on_update()
    tick_counter = tick_counter + 1
    
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
    
    if reply == "RESET" then
        Isaac.ExecuteCommand("restart")
        tick_counter = 0
    else
        parse_action(reply)
    end
end

-- 5. Input Injection Hook (Overrides both digital keyboard booleans and analog stick floats)
function IsaacRL:on_input(entity, hook, action)
    if not connected or entity == nil or entity.Type ~= EntityType.ENTITY_PLAYER then
        return nil
    end

    -- Helper to map current_action state to a float [0.0, 1.0] for any ButtonAction
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

    -- If the action isn't one of our 8 movement/shooting inputs, let the game handle it normally
    if val == nil then
        return nil
    end

    -- Return boolean for digital key checks (required for shooting!)
    if hook == InputHook.IS_ACTION_PRESSED or hook == InputHook.IS_ACTION_TRIGGERED then
        return val > 0.5
    -- Return float for analog stick checks (required for smooth movement)
    elseif hook == InputHook.GET_ACTION_VALUE then
        return val
    end
end

IsaacRL:AddCallback(ModCallbacks.MC_POST_UPDATE, IsaacRL.on_update)
IsaacRL:AddCallback(ModCallbacks.MC_INPUT_ACTION, IsaacRL.on_input)