print("homework 1 python script")


import os
import matplotlib.pyplot as plt

from models.GAN import GAN
from utils.loaders import load_safari

SECTION = 'gan'
RUN_ID = '0001'
DATA_NAME = 'camel'
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

mode =  'build' #'load' #



print("something else, I changed my file")

print("more updates")




import nest_asyncio
nest_asyncio.apply()
import discord

client = discord.Client()

GUILD= "GUILDNAME"        #{your-guild-name}
TOKEN = "OTExMzI0NTU5MTkxxxxxxxx"    # you should copy your token for your bot 


@client.event
async def on_ready():
    for guild in client.guilds:
        if guild.name == GUILD:
            break
    print(
        f'{client.user} is connected to the following guild:\n'
        f'{guild.name}(id: {guild.id})'
    )    


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    bot_quotes = [ 'Hello, I am a bot, 💯 emoji.','Bingo!',( 'happy, cool, happy, cool, happy, cool ',
            'no doubt no doubt no doubt no doubt.'),
    ]
    if message.content == 'wow!':
        response = random.choice(bot_quotes)
        await message.channel.send(response)
    
    if message.content == 'generateVAEIMG':
        ......


        fig, axs = plt.subplots(r, c, figsize=(15,15))
        cnt = 0

        for i in range(r):
            for j in range(c):
                c_diff = 99999
                c_img = None
                for k_idx, k in enumerate((x_train + 1) * 0.5):
                    
                    diff = compare_images(gen_imgs[cnt, :,:,:], k)
                    if diff < c_diff:
                        c_img = np.copy(k)
                        c_diff = diff
                axs[i,j].imshow(c_img, cmap = 'gray_r')
                axs[i,j].axis('off')
                cnt += 1

        fig.savefig(os.path.join(RUN_FOLDER, "images/sample_closest.png"))
        plt.close()


        await message.channel.send(filePATH)





client.run(TOKEN)
