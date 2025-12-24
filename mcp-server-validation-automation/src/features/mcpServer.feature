Feature: Connect MCP servers on Obot

    Background: Navigate to Obot
        Given I setup context for assertion
        When User navigates to the Obot main login page
        Then User opens chat Obot
        And User creates a new Project with no existing connections

    Scenario Outline: Validate <ServerName> MCP Server sequential prompts on Obot
        When User opens the MCP connector page
        And User selects "<ServerName>" MCP server
        And User selects "Connect To Server" button
        And User connects to the "<ConnectionName>" MCP server
        When User sends prompts to Obot AI chat for "<PromptName>" MCP server
        Then All prompt results should be validated and a report generated for the selected "<ReportName>" MCP server

        Examples:
            | ServerName                       | ConnectionName                     | PromptName       | ReportName       |
            | test-antv_charts                 | AntV Charts                        | AntV Charts      | AntV Charts      |
            | test-aws                         | AWS API                            | AWS API          | AWS API          |
            | test-aws_cdk                     | AWS CDK                            | AWS CDK          | AWS CDK          |
            | test-aws_documentation           | AWS Documentation                  | AWS Documentation| AWS Documentation|
            | test-aws_eks                     | AWS EKS                            | AWS EKS          | AWS EKS          |
            | test-aws_kendra                  | AWS Kendra                         | AWS Kendra       | AWS Kendra       |
            | test-aws_knowledge               | AWS Knowledge                      | AWS Knowledge    | AWS Knowledge    |
            | test-aws_redshift                | AWS Redshift                       | AWS Redshift     | AWS Redshift     |
            | test-bigquery                    | BigQuery                           | BigQuery Toolbox | BigQuery Toolbox |
            | test-brave_search                | Brave Search                       | Brave Search     | Brave Search     |
            | test-chroma                      | Chroma Cloud                       | Chroma Cloud     | Chroma Cloud     |
            | test-context7                    | Context7                           | Context7         | Context7         |
            # | test-datadog                     | Datadog                            | Datadog          | Datadog          |
            | test-databricks_genie            | Databricks Genie Space             | Databrick Genie  | Databrick Genie  |
            | test-databricks_uc_functions     | Databricks Unity Catalog Functions | Databrick Unity  | Databrick Unity  |
            | test-databricks_vector_search    | Databricks Vector Space            | Databrick Vector | Databrick Vector |
            | test-deepwiki                    | DeepWiki                           | DeepWiki         | DeepWiki         |
            | test-duckduckgo_search           | DuckDuckGo Search                  | DuckDuckGo Search| DuckDuckGo Search|
            | test-exa_search                  | Exa Search                         | Exa Search       | Exa Search       |
            | test-firecrawl                   | Firecrawl                          | Firecrawl        | Firecrawl        |
            | test-git lab                     | GitLab                             | GitLab           | GitLab           |
            | test-gitmcp                      | GitMCP                             | GitMCP           | GitMCP           |
            | test-google_cloud_run            | Google Cloud Run                   | Google Cloud Run | Google Cloud Run |
            | test-grafana                     | Grafana                            | Grafana          | Grafana          |
            | test-markitdown                  | MarkItDown                         | MarkItDown       | MarkItDown       |
            | test-microsoft-docs              | Microsoft Learn                    | Microsoft Learn  | Microsoft Learn  |
            # | test-postman                     | Postman                            | Postman          | Postman          |
            | test-pagerduty                   | PagerDuty                          | PagerDuty        | PagerDuty        |
            | test-redis                       | Redis                              | Redis            | Redis            |
            | test-salesforce                  | Salesforce                         | Salesforce       | Salesforce       |
            | test-slack                       | Slack                              | Slack            | Slack            |
            | test-tavily_search               | Tavily Search                      | Tavily Search    | Tavily Search    |
            | test-wordpress                   | WordPress                          | WordPress        | WordPress        |
            | test-ref                         | Ref                                | Ref              | Ref              |
            | test-render                      | Render                             | Render           | Render           |