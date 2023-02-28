# !/usr/bin/python3
# -*-encoding=utf-8-*-
# @author   : WangTianyu
# @Time     : 2022/5/27 16:44:16
# @File     : redisDB.py
import pickle
from redis import StrictRedis
redisDb = StrictRedis(host=redisDataSource[0], port=redisDataSource[1], db=redisDataSource[2], password=redisDataSource[3])


class RealTimeData(Resource):
    def _query(self):
        job_id, live_date = g.request.get("job_id"), cDate(g.request.get("live_date"))
        # 优先取redis数据,取出大于等于给定时间的数据。如果redis里没有就说明没数据进来。直接返回
        dt = getRedis(job_id + "_realtime_" + "*", toJs1970(live_date))
        return getResponse(dt)


def getRedis(k_match, live_date: int):
    # 检索满足key的键
    _redis = _rds.scan_iter(k_match)
    ks = []
    for k in _redis:
        if int(k.decode().split("_")[2]) > live_date:
            ks.append(k)
    # 排序
    ks.sort()
    # 取值
    dt = [pickle.loads(_rds.get(k)) for k in ks]
    return dt

