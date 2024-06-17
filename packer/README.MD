# OS base images | GCP

https://cloud.google.com/compute/docs/images/os-details

1.  Install googlecompute plugin

    ```bash
    packer plugins install github.com/hashicorp/googlecompute
    ```

2.  Run the following command for building the image

    ```bash
    packer build pow_worker.json
    ```
