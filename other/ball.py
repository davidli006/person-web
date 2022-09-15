"""
@DATE: 2022/9/15
@Author  : ld
"""
import datetime
import random
import time
import warnings
from io import StringIO

import numpy as np
import pandas as pd
import requests
import schedule
import torch
from sklearn import preprocessing

warnings.filterwarnings("ignore")


def isocalendar(dt):
    if isinstance(dt, str):
        dt = datetime.datetime.fromisoformat(dt)
    res = dt.isocalendar()
    if isinstance(res, tuple):
        return res[1]
    else:
        return res.week


class Ball(object):

    def __init__(self):
        self.inputs = None
        self.hidden_size = 128
        self.output_size = 1
        self.batch_size = 16
        self.base_col = ["new_sum", "new_mean", "new_qua", "new_std",
                         "new_min", "new_max", "new_skew", "new_kurt"]
        self.dt_col = ["no", "year", "month", "day", "week", "wd" ]
        self.select_ball = ["red_1", "red_2", "red_3", "red_4", "red_5", "red_6", "blue"]
        self.cost = torch.nn.L1Loss(reduction='mean')
        self.new_df = pd.DataFrame(columns=self.base_col+self.dt_col)


    def get_history(self):
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'}
        res = requests.get("https://e.17500.cn/getData/ssq.TXT", headers=headers)
        df = pd.read_csv(StringIO(res.text), usecols=[i for i in range(14)], sep=" ")
        df.columns = ["no", "date", "red_1", "red_2", "red_3", "red_4", "red_5", "red_6", "blue", "red_r_1", "red_r_2",
                      "red_r_3", "red_r_4", "red_r_5"]

        df = df[["no", "date", "red_1", "red_2", "red_3", "red_4", "red_5", "red_6", "blue"]]
        df["date"] = pd.to_datetime(df.date)

        df = df.assign(year=df.date.dt.year, month=df.date.dt.month, day=df.date.dt.day,
                       week= df.date.map(isocalendar), wd=df.date.dt.weekday).drop(columns="date")

        # 收集特征值
        df_ = df[["red_1", "red_2", "red_3", "red_4", "red_5", "red_6"]]
        df = df.assign(new_sum=df_.sum(axis=1), new_mean=df_.mean(axis=1), new_qua=df_.quantile(axis=1),
                       new_std=df_.std(axis=1), new_min=df_.min(axis=1), new_max=df_.max(axis=1),
                       new_skew=df_.skew(axis=1),  # 偏度
                       new_kurt=df_.kurt(axis=1),  # 锋度
                    )
        self.new_df = self.next_no(df.no.values[-1]).merge(df.mean().to_frame().T[self.base_col],
                                                      left_index=True, right_index=True)
        return df

    def next_no(self, last_no):
        dt = datetime.date.today()
        res = {"no": last_no + 1}
        res["year"] = dt.year
        res["month"] = dt.month
        res["day"] = dt.day
        res["week"] = isocalendar(dt)
        res["wd"] = dt.weekday()
        return pd.DataFrame([res])

    def train(self, inputs, ball):
        my_nn = torch.nn.Sequential(
            torch.nn.Linear(inputs.shape[1], self.hidden_size),
            torch.nn.Sigmoid(),
            torch.nn.Linear(self.hidden_size, self.output_size)
        )
        cost = torch.nn.L1Loss(reduction='mean')
        optimizer = torch.optim.Adam(my_nn.parameters(), lr=0.001)

        for i in range(10):
            length = len(inputs)
            for start in range(0, length, self.batch_size):
                end = start + self.batch_size if start + self.batch_size < length else length
                xx = torch.tensor(inputs[start: end], dtype=torch.float, requires_grad=True)
                yy = torch.tensor(ball[start:end], dtype=torch.float, requires_grad=True)

                prediction = my_nn(xx)
                loss = cost(prediction, yy)

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
        return my_nn

    def get_new_ball(self, ball_list):
        my_nn = self.train(self.inputs, ball_list)
        x = torch.tensor(self.fit_trans(self.new_df), dtype=torch.float)
        y = my_nn(x).data.numpy()
        return y

    def fit_trans(self, df_):
        return preprocessing.StandardScaler().fit_transform(df_)


    def run(self):
        df = self.get_history()
        last = df.tail(1).to_dict("records")[-1]

        df = pd.get_dummies(df)
        self.inputs = self.fit_trans(df[self.dt_col+self.base_col])

        new_pre = []
        for ball in self.select_ball:
            ball_list = np.array(df[ball])
            res = self.get_new_ball(ball_list)
            new_pre.append(res[0][0])

        res_pre = []
        rate = np.hstack([np.linspace(random.uniform(0.55, 0.99), 1, 3),
                          np.linspace(1, random.uniform(1.001, 1.1), 3), 1])
        for i, k in enumerate(new_pre):
            res_pre.append(int(k*rate[i]))

        self.send_msg(last, res_pre)

    def send_msg(self, last: dict, pre: list):
        """ 发送消息 """
        msg_before = f"日期: {last.get('year')}-{last.get('month')}-{last.get('day')} \n" \
                     f"期号: {last.get('no')} \n" \
                     f"开奖号码: {last.get('red_1')}-{last.get('red_2')}-{last.get('red_3')}-{last.get('red_4')}-" \
                     f"{last.get('red_5')}-{last.get('red_6')} \t <strong>{last.get('blue')}</strong> \n"
        blue = pre.pop(-1)
        pre = [str(i) for i in pre]
        msg_next = f"预测号码: {'-'.join(pre)} \t <strong>{blue}</strong>"
        data = {
            "token": "0498955a708d77",
            "title": "双色球信息",
            "content": f"{msg_before}\n\n{msg_next}",
        }
        requests.post("http://www.pushplus.plus/send/", data=data)


schedule.every().monday.at("10:00").do(Ball().run)
schedule.every().wednesday.at("10:00").do(Ball().run)
schedule.every().friday.at("10:00").do(Ball().run)
while True:
    schedule.run_pending()
    time.sleep(1)



