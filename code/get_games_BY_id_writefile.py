#coding:utf-8

import json
import requests
import time

# max_match_id = 999999999999999999   # 设置一个极大值作为match_id，可以查出最近的比赛(即match_id最大的比赛)。
max_match_id = 5072713911
target_match_num = 10000
lowest_mmr = 4000  # 匹配定位线，筛选该分数段以上的天梯比赛 

# url = "https://api.opendota.com/api/matches/5075751801?api_key=YOUR-API-KEY"
base_url = 'https://api.opendota.com/api/publicMatches?less_than_match_id='
session = requests.Session()
session.headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'
}

def crawl(input_url):
    time.sleep(0.1)   # 暂停一秒，防止请求过快导致网站封禁。
    crawl_tag = 0
    while crawl_tag==0:
        try:
            session.get("http://www.opendota.com/")  #获取了网站的cookie
            content = session.get(input_url)
            crawl_tag = 1
        except:
            print(u"Poor internet connection. We'll have another try.")
    json_content = json.loads(content.text)
    return json_content

match_list = []
recurrent_times = 0
write_tag = 0
with open('../data/matches_list_ranking.csv','w',encoding='utf-8') as fout:
    fout.write('比赛ID, 时间, 天辉英雄, 夜魇英雄, 天辉是否胜利\n')
    while(len(match_list)<target_match_num):
        json_content = crawl(base_url+str(max_match_id))
        for i in range(len(json_content)):
            match_id = json_content[i]['match_id']
            radiant_win = json_content[i]['radiant_win']
            start_time = json_content[i]['start_time']
            avg_mmr = json_content[i]['avg_mmr']
            if avg_mmr==None:
                avg_mmr = 0
            lobby_type = json_content[i]['lobby_type']
            game_mode = json_content[i]['game_mode']
            radiant_team = json_content[i]['radiant_team']
            dire_team = json_content[i]['dire_team']
            duration = json_content[i]['duration']  # 比赛持续时间
            if int(avg_mmr)<lowest_mmr:  # 匹配等级过低，忽略
                continue
            if int(duration)<900:   # 比赛时间过短，小于15min，视作有人掉线，忽略。
                continue
            if int(lobby_type)!=7 or (int(game_mode)!=3 and int(game_mode)!=22):
                continue
            x = time.localtime(int(start_time))
            game_time = time.strftime('%Y-%m-%d %H:%M:%S',x)
            one_game = [game_time,radiant_team,dire_team,radiant_win,match_id]
            match_list.append(one_game)
        max_match_id = json_content[-1]['match_id']
        recurrent_times += 1
        print(recurrent_times,len(match_list),max_match_id)
        if len(match_list)>target_match_num:
            match_list = match_list[:target_match_num]
        if write_tag<len(match_list):   # 如果小于新的比赛列表长度，则将新比赛写入文件
            for i in range(len(match_list))[write_tag:]:
                fout.write(str(match_list[i][4])+', '+match_list[i][0]+', '+match_list[i][1]+', '+\
                    match_list[i][2]+', '+str(match_list[i][3])+'\n')
            write_tag = len(match_list)

# with open('../data/matches_list_ranking.csv','w',encoding='utf-8') as fout:
#     fout.write('比赛ID, 时间, 天辉英雄, 夜魇英雄, 天辉是否胜利\n')
#     for item in match_list:
#         fout.write(str(item[4])+', '+item[0]+', '+item[1]+', '+item[2]+', '+str(item[3])+'\n')
        







