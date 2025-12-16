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
            | test-wordpress                   | WordPress                          | WordPress        | WordPress        |
            | test-gitlab                      | GitLab                             | GitLab           | GitLab           |
            | test-bigquery                    | BigQuery                           | BigQuery Toolbox | BigQuery Toolbox |
            # | test-datadog                     | Datadog                            | Datadog          | Datadog          |
            | test-databricks_uc_functions     | Databricks Unity Catalog Functions | Databrick Unity  | Databrick Unity  |
            | test-databricks_genie            | Databricks Genie Space             | Databrick Genie  | Databrick Genie  |
            | test-databricks_vector_search    | Databricks Vector Space            | Databrick Vector | Databrick Vector |
            | test-brave_search                | Brave Search                       | Brave Search     | Brave Search     |
            | test-chroma                      | Chroma Cloud                       | Chroma Cloud     | Chroma Cloud     |
            | test-firecrawl                   | Firecrawl                          | Firecrawl        | Firecrawl        |
            | test-gitmcp                      | GitMCP                             | GitMCP           | GitMCP           |
            | test-redis                       | Redis                              | Redis            | Redis            |
            # | test-postman                     | Postman                            | Postman          | Postman          |
            | test-tavily_search               | Tavily Search                      | Tavily Search    | Tavily Search    |
            | test-exa_search                  | Exa Search                         | Exa Search       | Exa Search       |
            | test-deepwiki                    | DeepWiki                           | DeepWiki         | DeepWiki         |
            | test-aws                         | AWS API                            | AWS API          | AWS API          |
            | test-aws_cdk                     | AWS CDK                            | AWS CDK          | AWS CDK          |
            | test-aws_documentation           | AWS Documentation                  | AWS Documentation| AWS Documentation|
            | test-aws_eks                     | AWS EKS                            | AWS EKS          | AWS EKS          |
            | test-aws_kendra                  | AWS Kendra                         | AWS Kendra       | AWS Kendra       |
            | test-aws_knowledge               | AWS Knowledge                      | AWS Knowledge    | AWS Knowledge    |
            | test-aws_redshift                | AWS Redshift                       | AWS Redshift     | AWS Redshift     |
            | test-context7                    | Context7                           | Context7         | Context7         |
            | test-paperduty                   | PaperDuty                          | PaperDuty        | PaperDuty        |
            | test-markitdown                  | MarkItDown                         | MarkItDown       | MarkItDown       |
            | test-microsoft-docs              | Microsoft Learn                    | Microsoft Learn  | Microsoft Learn  |
            | test-duckduckgo_search           | DuckDuckGo Search                  | DuckDuckGo Search| DuckDuckGo Search|