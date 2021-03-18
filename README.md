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
    "classes": ["person", "bag", "person with a bag", "woman riding a horse", "woman with a bag", "woman with black shirt and a bag"]
}'
```

The response looks like this:

```shell
[
    {
        "url": "https://images.unsplash.com/photo-1597308680537-1ba44407ffc0?ixid=MXwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHw%3D&ixlib=rb-1.2.1&auto=format&fit=crop&w=1834&q=80", 
        "labels": [
            "woman with black shirt and a bag", 
            "woman with a bag", 
            "person with a bag", 
            "bag", "person"
        ], 
        "probs": [1.0, 1.7488513970320696e-09, 1.1663764917350243e-19, 4.179975909038141e-30, 3.77612043676229e-30]
    }, 
    {
        "url": "https://images.unsplash.com/photo-1589270216117-7972b3082c7d?ixid=MXwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHw%3D&ixlib=rb-1.2.1&auto=format&fit=crop&w=1834&q=80", 
        "labels": [
            "person with a bag", 
            "woman with black shirt and a bag", 
            "bag", 
            "woman with a bag", 
            "person"
        ], 
        "probs": [1.0, 2.4879632576357835e-08, 2.065714813830402e-13, 7.658033346455602e-15, 1.1307645811408335e-23]
    }
]
```

### SageMaker Integration

This container is compatible with SageMaker so you should be able to host it as a SageMaker endpoint with no modification. 
The code supports for GPU and CPU instances.