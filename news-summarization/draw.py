import matplotlib.pyplot as plt
epoch = [0, 5, 10, 15, 20, 25]
rouge_1 = [
   0.1340354425975627,  
   0.1819593706898393,
   0.21077408463738082,
   0.22663403132407983,
   0.23678523731956616,
   0.2478703641368725
]
rouge_2 = [
    0.03817659718319515,
    0.06808203198031386,
    0.08205402018170908,
    0.08796661828103745,
    0.09216684856019464,
    0.09846630429921285
]
rouge_L = [
    0.12603324401616664,
    0.17000857486355228,
    0.1944794541322111,
    0.20609488263924353,
    0.2138624520518682,
    0.22389077191256973
]
plt.plot(epoch, [i * 100 for i in rouge_1], label='rouge_1')
plt.plot(epoch, [i * 100 for i in rouge_2], label='rouge_2')
plt.plot(epoch, [i * 100 for i in rouge_L], label='rouge_L')
plt.title('Rouge score')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('score')
# add line at y = 22.0, y = 8.5, y = 20.5
plt.axhline(y=22.0, color='r', linestyle='--')
# add text on the line at y = 22.0
plt.text(0, 22.0, 'rouge-1 baseline', fontsize=10)
plt.axhline(y=8.5, color='r', linestyle='--')
plt.text(0, 8.5, 'rouge-2 baseline', fontsize=10)
plt.axhline(y=20.5, color='r', linestyle='--')
plt.text(0, 20.5, 'rouge-L baseline', fontsize=10)
plt.savefig('./outputs/rouge.png')