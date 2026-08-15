"""Certificate-core errors. These are execution failures, not scientific dispositions."""
from __future__ import annotations


class CertificateError(Exception):
    """Base class for certificate-core failures."""


class InvalidGateStateError(CertificateError):
    """A gate state is unknown or invalid for scientific classification."""


class LineageError(CertificateError):
    """Revision-tree or scope-mutation violation."""


class ArtifactExistsError(CertificateError):
    """Create-only artifact path already exists."""


class DuplicateCertificateIdError(CertificateError):
    """A certificate ID has already been issued."""


class PreregistrationError(CertificateError):
    """Preregistration commit is missing, dirty, or inconsistent."""


class SchemaError(CertificateError):
    """Spec or record failed schema validation."""
