import requests

GHOST_API_URL='https://www.langchain.ca/blog/ghost/api/v3/content/posts/'

# Your Ghost API key
GHOST_API_KEY='65f1ac9109333f517ddaac03:2cb16de1e79fa7d0e140896c14b541b08de4b54507b79ed54ca1e751b62de6a4'




def post_blog(title, content):
    payload = {
        'posts': [{
            'title': title,
            'html': content,
            'tags': ['tag1', 'tag2'],
            'status': 'draft'
        }]
    }
    print(payload)
    response = requests.post(GHOST_API_URL, json=payload, headers={'Authorization': f'Ghost {GHOST_API_KEY}'}, timeout=60)

    # Check response status
    if response.status_code == 201:
        print('Blog post published successfully!')
    else:
        print(f'Failed to publish blog post. Status code: {response.status_code}, Error: {response.text}')
