# https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/google_project_service
# Habilita servicios de GCP

resource "google_project_service" "compute" {
  service = "compute.googleapis.com"
  disable_dependent_services = true
  disable_on_destroy = false
}

resource "google_project_service" "container" {
  service = "container.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloud_resource_manager" {
  service            = "cloudresourcemanager.googleapis.com"
  disable_on_destroy = false
}