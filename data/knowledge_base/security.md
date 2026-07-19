# Security Knowledge Base

## Encryption at rest and in transit
All customer data is encrypted at rest using AES-256 and in transit using TLS 1.2+.
Encryption keys are managed in a FIPS 140-2 validated KMS with automatic rotation
every 90 days. Field-level encryption is available for sensitive attributes.

## Multi-factor authentication (MFA)
MFA is enforced for all administrative and privileged accounts. We support TOTP
authenticator apps, WebAuthn / FIDO2 hardware keys, and SSO-enforced MFA. MFA can
be made mandatory for all end users via an org-level policy.

## Penetration testing and vulnerability management
We engage an independent third party to perform full-scope penetration tests at
least annually, plus after major architecture changes. Continuous automated
vulnerability scanning runs weekly, and critical findings are remediated within
7 days per our vulnerability management SLA.

## Security incident response
We operate a 24/7 security incident response process aligned to NIST 800-61.
Confirmed incidents affecting customer data are communicated to affected customers
without undue delay and within 72 hours of confirmation, including scope and
remediation status.

## Endpoint and infrastructure security
Production access requires MFA and is brokered through a bastion with full session
logging. All endpoints run managed EDR, and infrastructure changes go through
peer-reviewed, audited CI/CD pipelines.
