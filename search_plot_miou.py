# ===============================
# author : Jingbo Lin
# contact: ljbxd180612@gmail.com
# github : github.com/mrluin
# ===============================

import json
import matplotlib.pyplot as plt

from matplotlib.pyplot import MultipleLocator

f1 = open('./0-100-005-0025-0003-1-noentropy-nop-search.json')
# miou window_3810e671e81e28
json_dict1 = json.load(f1)
f1.close()
#print(json_dict['jsons']['window_380ccf45caf6f6']['title'])
x1 = json_dict1['jsons']['window_3810e671e81e28']['content']['data'][0]['x']
y1 = json_dict1['jsons']['window_3810e671e81e28']['content']['data'][0]['y']

f2 = open('./warm20-epochs100-wlr005-slr0025-alr0003-freq1-noentropy-search.json')
#['window_380c721b737d1a', 'window_380c721b703dbe', 'window_380c721b726022', 'window_380c721b6f9272', 'window_380c721b73f400', 'window_380c721b71e4fa', 'window_380c721b715dce', 'window_380c721b70d1b2', 'window_380c721b6ede3e']
# warmup_loss               train_search_miou        entropy                  train_search_accuracy    warmup_miou              network_entropy          cell_entropy             train_search_f1socre     train_search_loss
json_dict2 = json.load(f2)
f2.close()
x2 = json_dict2['jsons']['window_380c721b703dbe']['content']['data'][0]['x']
y2 = json_dict2['jsons']['window_380c721b703dbe']['content']['data'][0]['y']
#print(json_dict2['jsons']['window_380c721b6ede3e']['title'])

f3 = open('./20-100-005-0025-0003-1-noentropy-nop-search.json')
# loss window_380ff39b04ce44 # miou window_380ff39b062c38
json_dict3 = json.load(f3)
f3.close()
x3 = json_dict3['jsons']['window_380ff39b062c38']['content']['data'][0]['x']
y3 = json_dict3['jsons']['window_380ff39b062c38']['content']['data'][0]['y']

f4 = open('./20-100-005-0025-0003-0006003-linear-search.json')
# miou window window_3811bb5b3e3338
json_dict4 = json.load(f4)
f4.close()
x4 = json_dict4['jsons']['window_3811bb5b3e3338']['content']['data'][0]['x']
y4 = json_dict4['jsons']['window_3811bb5b3e3338']['content']['data'][0]['y']


plt.title('searching miou')
plt.xlabel('epochs')
plt.ylabel('miou')

plt.plot(x1, y1, label='BdnasNet-nude', linewidth=1.5, linestyle='-', color=(255/255, 127/255, 0/255))
plt.plot(x2, y2, label='BdnasNet-WP', linewidth=1.5, linestyle='-', color='green')
plt.plot(x3, y3, label='BdnasNet-W', linewidth=1.5, linestyle='-', color='blue')
plt.plot(x4, y4, label='BdnasNet', linewidth=1.5, linestyle='-', color='red')

plt.legend(loc='upper left')
ax = plt.gca()
x_major_locator = MultipleLocator(20)
ax.xaxis.set_major_locator(x_major_locator)
plt.xlim(0, 200)
plt.grid(color='grey', linestyle=':')
plt.xscale('linear')
plt.yscale('linear')
plt.autoscale()
plt.show()