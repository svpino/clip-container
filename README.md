# OpenAI's CLIP — REST API

This is a container wrapping OpenAI's CLIP model in a RESTful interface.

### Running the container locally

First, build the container:

```shell
docker build -t clip-container:latest .
```

Then, you can run it:

```shell
docker run -it -p 8080:8080 --name "clip-container" --rm clip-container:latest /opt/ml/code/serve
```

### Sending requests:

The container exposes two different endpoints:

- `GET /ping`: Returns 200 status if the container is working properly.
- `POST /invocations`: Processes a list of images and returns the list of labels with their corresponding probabilities.

Here is an example request assuming the container is listening in port `8080`:

```shell
curl --location --request POST 'http://localhost:8080/invocations' \
--header 'Content-Type: application/json' \
--data-raw '{
    "images": [
        "https://images.unsplash.com/photo-1597308680537-1ba44407ffc0?ixid=MXwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHw%3D&ixlib=rb-1.2.1&auto=format&fit=crop&w=1834&q=80",
        "https://images.unsplash.com/photo-1589270216117-7972b3082c7d?ixid=MXwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHw%3D&ixlib=rb-1.2.1&auto=format&fit=crop&w=1834&q=80"],
    "classes": [["person", 1], ["bag", 0], ["person with a bag", 1], ["woman riding a horse", 1], ["woman with a bag", 1], ["woman with black shirt and a bag", 1]]
}'
```

The response looks like this:

```shell
[
    {
        "classification": 1,
        "confidence": 0.8420763611793518,
        "prompts": [
            [
                5,
                4,
                2
            ],
            [
                "woman with black shirt and a bag",
                "woman with a bag",
                "person with a bag"
            ],
            [
                0.8420763611793518,
                0.13471820950508118,
                0.020683519542217255
            ]
        ]
    },
    {
        "classification": 1,
        "confidence": 0.8793354034423828,
        "prompts": [
            [
                2,
                5,
                4
            ],
            [
                "person with a bag",
                "woman with black shirt and a bag",
                "woman with a bag"
            ],
            [
                0.8793354034423828,
                0.06026090309023857,
                0.027413571253418922
            ]
        ]
    }
]
```

### SageMaker Integration

This container is compatible with SageMaker so you should be able to host it as a SageMaker endpoint with no modifications. 
The code supports GPU and CPU instances.
