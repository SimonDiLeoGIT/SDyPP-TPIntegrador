# https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/compute_router

# Crea un router y lo asocia a la red main
resource "google_compute_router" "router" {
  name    = "router"
  region  = var.region
  network = google_compute_network.main.id
}