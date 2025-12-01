import json

from command.registry import command
from utils.logger import get_logger
from control.move import move_to
from utils.math import rad2deg

logger = get_logger("echo")

@command("ik_det")
async def ik_det_handler(websocket, state, *args):
    """
    逆解坐标值
    :param websocket: websocket 对象
    :param state: 全局共享状态
    :param args: 命令参数
    :return: none
    """

    det_r, det_theta, det_h = float(args[0]), float(args[1]), float(args[2])
    r, theta, h = state.current_pos
    r += det_r
    theta += det_theta
    h += det_h
    angles = move_to(r, theta, h)
    return {"type": "success", "args": rad2deg(angles)}