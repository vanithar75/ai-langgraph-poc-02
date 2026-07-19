# Product Knowledge Base

## Single sign-on (SSO)
We support SSO via SAML 2.0 and OpenID Connect (OIDC). SCIM 2.0 is available for
automated user provisioning and deprovisioning. Just-in-time provisioning is
supported for supported identity providers (Okta, Entra ID, Google Workspace).

## Role-based access control (RBAC)
The platform ships with granular RBAC: predefined roles (Owner, Admin, Editor,
Reviewer, Viewer) plus custom roles with per-resource permissions. Access can be
scoped by workspace, project, and environment.

## Uptime SLA and availability
Our standard enterprise SLA guarantees 99.9% monthly uptime with service credits
for missed targets. Multi-AZ deployment and automated failover keep RTO under
one hour and RPO under five minutes.

## APIs and integrations
We provide a documented REST API, webhooks, and prebuilt integrations for Slack,
Salesforce, Jira, and Google Workspace. All API access is authenticated with
scoped API keys or OAuth 2.0 and rate-limited per plan.

## Backups and disaster recovery
Encrypted backups are taken continuously and stored in geographically redundant
regions. Backups are tested via quarterly restore drills, and a documented DR
plan is exercised at least annually.
