import json

from command.registry import command
from utils.logger import get_logger
from control.move import set_angle
from control.kinematics import fk
from utils.math import rad2deg, deg2rad

logger = get_logger("echo")

@command("fk_det")
async def fk_det_handler(websocket, state, *args):
    """
    正解运动学 相对模式
    :param websocket: websocket 对象
    :param state: 全局共享状态
    :param args: 命令参数
    :return: none
    """

    # angle1, angle2, angle3 = float(args[0]), float(args[1]), float(args[2])
    r, theta, h = state.current_pos
    angle1, angle2, angle3 = fk(r, state, h)
    angle1, angle2, angle3 = deg2rad(angle1, angle2, angle3)

    det_angle1, det_angle2, det_angle3 = deg2rad((args[0], args[1], args[2]))
    angle1 += det_angle1
    angle2 += det_angle2
    angle3 += det_angle3
    set_angle(angle1, angle2, angle3)
    r, theta, h = fk(angle1, angle2, angle3)
    return {"type": "success", "args": [r, rad2deg(theta), h]}