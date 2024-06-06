terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">=4.60.0"
    }
  }

  backend "gcs" {
    bucket  = "terraform_state_cloud"
    prefix  = "cluster/state"
  }

  required_version = ">= 1.4.5"
}

provider "google" {
  credentials = file(var.credentials)
  project     = var.project
  region      = var.region
  zone        = var.zone
}

