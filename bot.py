import discord
import os
from dotenv import load_dotenv
from transformers import pipeline
from PIL import Image
import io
import asyncio

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
tree = discord.app_commands.CommandTree(client)

print("Loading model...")
try:
  sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
  )
  image_captioner = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-large"
  )
  print("ML models loaded successfully.")

except Exception as e:
  print(f"Error loading Models: {e}")
  exit()


@client.event
async def on_ready():
  await tree.sync()
  print(f'Logged in as {client.user} (ID: {client.user.id})\n')

@client.event
async def on_message(message: discord.Message):
  if message.author.bot:
    return
  
  if message.attachments:
    for attachment in message.attachments:
      if attachment.content_type in ['image/jpeg', 'imae/png', 'image/jpg', 'image/gif']:
        print(f"Image deteced in message from {message.author.name}. Analyzing...")

        async with message.channel.typing():
          try:
            image_bytes = await attachment.read()
            image = Image.open(io.BytesIO(image_bytes))

            loop = asyncio.get_event_loop()
            caption_result = await loop.run_in_executor(None, image_captioner, image)
            caption = caption_result[0]['generated_text']

            await message.reply(f"Image Caption: {caption}")

          except Exception as e:
            print(f"Error processing image: {e}")
            await message.reply("Sorry, I couldn't process the image.")
        break

@tree.command(name="analyze", description="Analyzes the sentiment of a piece of text.")
async def analyze_sentiment(interaction: discord.Interaction, text: str):
  await interaction.response.defer()

  try:
    result = sentiment_analyzer(text)[0]
    label = result['label'].captitalize()
    score = result['score']

    if label == 'Positive':
      emoji = '😊'
      color = discord.Color.green()
    elif label == 'Negative':
      emoji = '😞'
      color = discord.Color.red()
    else:   # neutral denote karega
      emoji = '😐'
      color = discord.Color.gold()

    embed = discord.Embed(
      title="Sentiment Analysis Result",
      color=color
      )
    embed.add_field(name="Input Text", value=f"```{text}```", inline=False)
    embed.add_field(name="Result", value=f"**{label}** ", inline=True)
    embed.add_field(name="Confidence", value=f"{score:.2%}, inline=True")

    await interaction.followup.send(embed=embed)

  except Exception as e:
    print(f"Error during sentiment analysis: {e}")
    await interaction.followup.send("Sorry, I couldn't analyze the sentiment of the text.")


client.run(TOKEN)